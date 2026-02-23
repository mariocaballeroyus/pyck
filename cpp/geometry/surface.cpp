#include "surface.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace pyck
{

// ─── helpers ────────────────────────────────────────────────────────────────

static std::vector<double> greville_abscissae(const BSpline& bs)
{
    std::size_t n = bs.num_basis();
    std::size_t p = bs.degree();
    const auto& T = bs.knots();
    std::vector<double> xi(n);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 1; j <= p; ++j)
            sum += T[i + j];
        xi[i] = sum / static_cast<double>(p);
    }
    return xi;
}

/// Coordinate label for the i-th local direction (0 → 'x', 1 → 'y').
static char coord_label(std::size_t i) { return (i == 0) ? 'x' : 'y'; }

// ─── factory methods ────────────────────────────────────────────────────────

SurfacePatch SurfacePatch::flat_plate(std::array<BSpline*, 2> bases,
                                       double L_u, double L_v)
{
    auto xi_u = greville_abscissae(*bases[0]);
    auto xi_v = greville_abscissae(*bases[1]);
    std::size_t n_u = bases[0]->num_basis();
    std::size_t n_v = bases[1]->num_basis();

    Eigen::MatrixXd P(n_u * n_v, 3);
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            P.row(a * n_v + b) << L_u * xi_u[a], L_v * xi_v[b], 0.0;

    return SurfacePatch(3, {bases[0], bases[1]}, P);
}

SurfacePatch SurfacePatch::paraboloid(std::array<BSpline*, 2> bases)
{
    double x_ctrl[] = {0.0, 0.5, 1.0};
    double y_ctrl[] = {0.0, 0.5, 1.0};
    double zu_ctrl[] = {0.0, 0.0, 1.0};
    double zv_ctrl[] = {0.0, 0.0, 1.0};

    Eigen::MatrixXd P(9, 3);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            P.row(a * 3 + b) << x_ctrl[a], y_ctrl[b], zu_ctrl[a] + zv_ctrl[b];

    return SurfacePatch(3, {bases[0], bases[1]}, P);
}

// ─── eval ───────────────────────────────────────────────────────────────────

Eigen::MatrixXd SurfacePatch::eval(const Eigen::MatrixXd& params) const
{
    return tensor_product_.eval(params) * control_points_;
}

// ─── private parametric derivative helper ───────────────────────────────────

std::vector<std::vector<Eigen::MatrixXd>>
SurfacePatch::eval_parametric_derivs_(const Eigen::MatrixXd& params,
                                      std::size_t order) const
{
    std::vector<std::vector<Eigen::MatrixXd>> result(order + 1);
    for (std::size_t i = 0; i <= order; ++i)
        result[i].resize(order + 1 - i);

    if (order == 0) {
        result[0][0] = eval(params);
    } else {
        auto derivs = tensor_product_.eval_derivs(params, {order, order});
        for (std::size_t i = 0; i <= order; ++i)
            for (std::size_t j = 0; j <= order - i; ++j)
                result[i][j] = derivs[i * (order + 1) + j] * control_points_;
    }
    return result;
}

// ─── eval_physical_derivs (local tangent-plane coordinates) ─────────────────

std::unordered_map<std::string, Eigen::MatrixXd>
SurfacePatch::eval_physical_derivs(const Eigen::MatrixXd& params,
                                   std::size_t order) const
{
    order = std::max<std::size_t>(1, std::min<std::size_t>(order, 3));

    const Eigen::Index Q = params.rows();
    const std::size_t gd = gdim_;
    const std::size_t K  = tensor_product_.num_basis();

    // ── Parametric basis function derivatives ──
    auto raw = tensor_product_.eval_derivs(params, {order, order});
    // raw[i*(order+1)+j] = d^{i+j}R/(du^i dv^j),  shape (Q × K)
    auto R = [&](std::size_t i, std::size_t j) -> const Eigen::MatrixXd& {
        return raw[i * (order + 1) + j];
    };

    // ── Geometry covariant base vectors ──
    Eigen::MatrixXd g1 = R(1, 0) * control_points_;  // ∂x/∂u  (Q × gd)
    Eigen::MatrixXd g2 = R(0, 1) * control_points_;  // ∂x/∂v  (Q × gd)

    // ── Build local orthonormal frame & J_local (2×2 upper-triangular) ──
    // per quadrature point.  Also store J_local_inv.
    //   J_local = [[a, b], [0, d]]
    //   e1 = g1/|g1|,  e2 = (g2 - (g2·e1)e1) / |...|
    //   a = |g1|,  b = (g1·g2)/|g1|,  d = |g2 - b*e1|
    Eigen::VectorXd a_vec(Q), b_vec(Q), d_vec(Q);

    // J_local_inv rows: stored as (Q × 2) for each local direction
    // Jinv = [[1/a, -b/(a*d)], [0, 1/d]]
    Eigen::MatrixXd Jinv(Q, 4); // [Jinv_00, Jinv_01, Jinv_10, Jinv_11]

    for (Eigen::Index q = 0; q < Q; ++q) {
        Eigen::VectorXd g1q = g1.row(q).transpose();
        Eigen::VectorXd g2q = g2.row(q).transpose();
        double a = g1q.norm();
        double b_val = g1q.dot(g2q) / a;
        Eigen::VectorXd g2_perp = g2q - (b_val) * (g1q / a);
        double d_val = g2_perp.norm();

        a_vec(q) = a;
        b_vec(q) = b_val;
        d_vec(q) = d_val;

        Jinv(q, 0) = 1.0 / a;
        Jinv(q, 1) = -b_val / (a * d_val);
        Jinv(q, 2) = 0.0;
        Jinv(q, 3) = 1.0 / d_val;
    }

    // ── Result map ──
    std::unordered_map<std::string, Eigen::MatrixXd> result;
    result["N"] = R(0, 0);

    // ── 1st order: dN/dx_i = Jinv_{α,i} dN/dξ_α ──
    // i ∈ {0,1} → local coords (x, y on the surface)
    // α ∈ {0=u, 1=v}
    Eigen::MatrixXd dNdx1(Q, K), dNdx2(Q, K);
    for (Eigen::Index q = 0; q < Q; ++q) {
        // dN/dx = Jinv[0,0]*dN/du + Jinv[1,0]*dN/dv  (column 0 of Jinv^T)
        //       = (1/a)*dN/du + 0*dN/dv
        // dN/dy = Jinv[0,1]*dN/du + Jinv[1,1]*dN/dv  (column 1 of Jinv^T)
        //       = (-b/(ad))*dN/du + (1/d)*dN/dv
        // Jinv_{α,i} means row α col i, so:
        // dN/dx_i = sum_α Jinv_{α,i} * dN/dξ_α
        dNdx1.row(q) = Jinv(q, 0) * R(1, 0).row(q) + Jinv(q, 2) * R(0, 1).row(q);
        dNdx2.row(q) = Jinv(q, 1) * R(1, 0).row(q) + Jinv(q, 3) * R(0, 1).row(q);
    }
    result["dNdx"] = dNdx1;
    result["dNdy"] = dNdx2;

    if (order < 2) return result;

    // ── Geometry 2nd derivatives (for Christoffel corrections) ──
    Eigen::MatrixXd x_11 = R(2, 0) * control_points_;  // ∂²x/∂u²     (Q × gd)
    Eigen::MatrixXd x_12 = R(1, 1) * control_points_;  // ∂²x/∂u∂v    (Q × gd)
    Eigen::MatrixXd x_22 = R(0, 2) * control_points_;  // ∂²x/∂v²     (Q × gd)

    // Compute ∂J_local/∂ξ_β for β=0,1 at each q-point.
    // J_local = [[a, b], [0, d]]  with  a=|g1|,  b=(g1·g2)/a,  d=|g2_perp|
    // We need da/dξ_β, db/dξ_β, dd/dξ_β
    // dg1/dξ_0 = x_{,11}, dg1/dξ_1 = x_{,12}
    // dg2/dξ_0 = x_{,12}, dg2/dξ_1 = x_{,22}

    // Store dJloc/dξ_β as 4 values [dJloc_00, dJloc_01, dJloc_10=0, dJloc_11]
    // dJloc_beta[β] shape (Q × 4)
    Eigen::MatrixXd dJloc_0(Q, 4), dJloc_1(Q, 4);

    // Also compute dJinv/dξ_β = -Jinv * (dJloc/dξ_β) * Jinv
    Eigen::MatrixXd dJinv_0(Q, 4), dJinv_1(Q, 4);

    for (Eigen::Index q = 0; q < Q; ++q) {
        double a = a_vec(q), bv = b_vec(q), dv = d_vec(q);
        Eigen::VectorXd g1q = g1.row(q).transpose();
        Eigen::VectorXd g2q = g2.row(q).transpose();

        // x_{,αβ} at this point
        Eigen::VectorXd x11 = x_11.row(q).transpose();
        Eigen::VectorXd x12 = x_12.row(q).transpose();
        Eigen::VectorXd x22 = x_22.row(q).transpose();

        // For β = 0 (derivative w.r.t. u): dg1/du = x11, dg2/du = x12
        // For β = 1 (derivative w.r.t. v): dg1/dv = x12, dg2/dv = x22
        Eigen::VectorXd dg1[2] = {x11, x12};
        Eigen::VectorXd dg2[2] = {x12, x22};

        Eigen::MatrixXd* dJloc_arr[2] = {&dJloc_0, &dJloc_1};
        Eigen::MatrixXd* dJinv_arr[2] = {&dJinv_0, &dJinv_1};

        for (int beta = 0; beta < 2; ++beta) {
            // da/dξ_β = (g1 · dg1[β]) / a
            double da = g1q.dot(dg1[beta]) / a;
            // db/dξ_β = [(dg1[β]·g2 + g1·dg2[β]) - bv*da] / a
            double db = (dg1[beta].dot(g2q) + g1q.dot(dg2[beta]) - bv * da) / a;
            // dd/dξ_β = [(g2·dg2[β]) - bv*db] / d
            double dd = (g2q.dot(dg2[beta]) - bv * db) / dv;

            (*dJloc_arr[beta])(q, 0) = da;
            (*dJloc_arr[beta])(q, 1) = db;
            (*dJloc_arr[beta])(q, 2) = 0.0;
            (*dJloc_arr[beta])(q, 3) = dd;

            // dJinv/dξ_β = -Jinv * dJloc * Jinv  (2×2 matrix multiply)
            Eigen::Matrix2d Ji, dJl;
            Ji << Jinv(q, 0), Jinv(q, 1), Jinv(q, 2), Jinv(q, 3);
            dJl << da, db, 0.0, dd;
            Eigen::Matrix2d dJi = -Ji * dJl * Ji;
            (*dJinv_arr[beta])(q, 0) = dJi(0, 0);
            (*dJinv_arr[beta])(q, 1) = dJi(0, 1);
            (*dJinv_arr[beta])(q, 2) = dJi(1, 0);
            (*dJinv_arr[beta])(q, 3) = dJi(1, 1);
        }
    }

    // ── 2nd order: d²N/(dx_i dx_j) ──
    // = Jinv_{α,i} Jinv_{β,j} d²N/(dξ_α dξ_β)
    //   + Σ_α [Σ_β Jinv_{β,j} dJinv_{α,i}/dξ_β] dN/dξ_α
    // Unique combos: (i,j) ∈ {(0,0),(0,1),(1,1)} → dNdxdx, dNdxdy, dNdydy
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = i; j < 2; ++j) {
            Eigen::MatrixXd d2N(Q, K);
            for (Eigen::Index q = 0; q < Q; ++q) {
                // Jinv entries: Jinv_{α,i} stored as Jinv(q, α*2+i)
                double Ji_0i = Jinv(q, 0 * 2 + i), Ji_1i = Jinv(q, 1 * 2 + i);
                double Ji_0j = Jinv(q, 0 * 2 + j), Ji_1j = Jinv(q, 1 * 2 + j);

                // Term 1: Jinv_{α,i} Jinv_{β,j} d²N/(dξ_α dξ_β)
                Eigen::RowVectorXd t1 =
                    Ji_0i * Ji_0j * R(2, 0).row(q)                    // du du
                  + (Ji_0i * Ji_1j + Ji_1i * Ji_0j) * R(1, 1).row(q) // du dv + dv du
                  + Ji_1i * Ji_1j * R(0, 2).row(q);                  // dv dv

                // Term 2: correction from varying Jinv
                // Γ^α_{ij} = Σ_β Jinv_{β,j} (dJinv_{α,i}/dξ_β)
                const Eigen::MatrixXd* dJi[2] = {&dJinv_0, &dJinv_1};
                double Gamma_u = 0.0, Gamma_v = 0.0;
                for (int beta = 0; beta < 2; ++beta) {
                    double Jibj = Jinv(q, beta * 2 + j);
                    Gamma_u += Jibj * (*dJi[beta])(q, 0 * 2 + i);
                    Gamma_v += Jibj * (*dJi[beta])(q, 1 * 2 + i);
                }

                Eigen::RowVectorXd t2 = Gamma_u * R(1, 0).row(q)
                                       + Gamma_v * R(0, 1).row(q);

                d2N.row(q) = t1 + t2;
            }

            std::string key = "dNd";
            key += coord_label(i);
            key += 'd';
            key += coord_label(j);
            result[key] = d2N;
        }
    }

    if (order < 3) return result;

    // ── 3rd-order geometry derivatives ──
    Eigen::MatrixXd x_111 = R(3, 0) * control_points_;
    Eigen::MatrixXd x_112 = R(2, 1) * control_points_;
    Eigen::MatrixXd x_122 = R(1, 2) * control_points_;
    Eigen::MatrixXd x_222 = R(0, 3) * control_points_;

    // Compute d²Jloc/dξ_β dξ_γ and d²Jinv/dξ_β dξ_γ
    // Stored as (Q × 4) for each (β,γ) combo: (0,0),(0,1),(1,1)
    // d2Jinv = -Jinv * d2Jloc * Jinv - dJinv_β * dJloc_γ * Jinv - Jinv * dJloc_β * dJinv_γ
    Eigen::MatrixXd d2Jinv[2][2];
    for (int b = 0; b < 2; ++b)
        for (int g = b; g < 2; ++g)
            d2Jinv[b][g] = Eigen::MatrixXd(Q, 4);

    for (Eigen::Index q = 0; q < Q; ++q) {
        double a = a_vec(q), bv = b_vec(q), dv = d_vec(q);
        Eigen::VectorXd g1q = g1.row(q).transpose();
        Eigen::VectorXd g2q = g2.row(q).transpose();

        // 2nd & 3rd geometry derivs
        Eigen::VectorXd x11 = x_11.row(q).transpose();
        Eigen::VectorXd x12 = x_12.row(q).transpose();
        Eigen::VectorXd x22 = x_22.row(q).transpose();
        Eigen::VectorXd dg1[2] = {x11, x12};
        Eigen::VectorXd dg2[2] = {x12, x22};

        // 3rd-order: d²g1/dξ_β dξ_γ and d²g2/dξ_β dξ_γ
        // d²g1/(du du) = x_{,111}, d²g1/(du dv) = x_{,112}, d²g1/(dv dv) = x_{,122}
        // d²g2/(du du) = x_{,112}, d²g2/(du dv) = x_{,122}, d²g2/(dv dv) = x_{,222}
        Eigen::VectorXd x111 = x_111.row(q).transpose();
        Eigen::VectorXd x112 = x_112.row(q).transpose();
        Eigen::VectorXd x122 = x_122.row(q).transpose();
        Eigen::VectorXd x222 = x_222.row(q).transpose();
        // d²g1[β][γ]: β+γ maps -> x_{1,β+γ+1}  but more carefully:
        // g1 = x_{,1}, so dg1/dξβ = x_{,1β}, d²g1/(dξβ dξγ) = x_{,1βγ}
        Eigen::VectorXd d2g1[2][2], d2g2[2][2];
        d2g1[0][0] = x111; d2g1[0][1] = d2g1[1][0] = x112; d2g1[1][1] = x122;
        d2g2[0][0] = x112; d2g2[0][1] = d2g2[1][0] = x122; d2g2[1][1] = x222;

        // Recompute da/dξ, db/dξ, dd/dξ for both β
        double da[2], db[2], dd[2];
        for (int beta = 0; beta < 2; ++beta) {
            da[beta] = g1q.dot(dg1[beta]) / a;
            db[beta] = (dg1[beta].dot(g2q) + g1q.dot(dg2[beta]) - bv * da[beta]) / a;
            dd[beta] = (g2q.dot(dg2[beta]) - bv * db[beta]) / dv;
        }

        Eigen::Matrix2d Ji;
        Ji << Jinv(q, 0), Jinv(q, 1), Jinv(q, 2), Jinv(q, 3);

        Eigen::Matrix2d dJl[2];
        dJl[0] << da[0], db[0], 0.0, dd[0];
        dJl[1] << da[1], db[1], 0.0, dd[1];

        for (int beta = 0; beta < 2; ++beta) {
            for (int gamma = beta; gamma < 2; ++gamma) {
                // d²a/(dξβ dξγ)
                double d2a = (dg1[beta].dot(dg1[gamma]) + g1q.dot(d2g1[beta][gamma])
                              - da[beta] * da[gamma]) / a;
                double d2b = (d2g1[beta][gamma].dot(g2q) + dg1[beta].dot(dg2[gamma])
                              + dg1[gamma].dot(dg2[beta]) + g1q.dot(d2g2[beta][gamma])
                              - db[beta] * da[gamma] - bv * d2a
                              - da[beta] * db[gamma]) / a
                           + (bv * da[beta] - dg1[beta].dot(g2q) - g1q.dot(dg2[beta])
                              + bv * da[beta]) * da[gamma] / (a * a);

                // b = (g1·g2)/a
                // db/dξ = [(dg1·g2 + g1·dg2) - b * da]/a
                // d²b/(dξβ dξγ) = d/dξγ { [(dg1β·g2 + g1·dg2β) - bv*daβ] / a }
                //  = { [d2g1βγ·g2 + dg1β·dg2γ + dg1γ·dg2β + g1·d2g2βγ
                //       - dbγ*daβ - bv*d2aβγ ] / a
                //     - [(dg1β·g2 + g1·dg2β) - bv*daβ] * daγ / a² }

                // Let me just recompute properly
                double num_b_beta = dg1[beta].dot(g2q) + g1q.dot(dg2[beta]) - bv * da[beta];
                double dnum_b = d2g1[beta][gamma].dot(g2q) + dg1[beta].dot(dg2[gamma])
                              + dg1[gamma].dot(dg2[beta]) + g1q.dot(d2g2[beta][gamma])
                              - db[gamma] * da[beta] - bv * d2a;
                d2b = (dnum_b * a - num_b_beta * da[gamma]) / (a * a);

                double num_d_beta = g2q.dot(dg2[beta]) - bv * db[beta];
                double dnum_d = dg2[gamma].dot(dg2[beta]) + g2q.dot(d2g2[beta][gamma])
                              - db[gamma] * db[beta] - bv * d2b;
                double d2d = (dnum_d * dv - num_d_beta * dd[gamma]) / (dv * dv);

                Eigen::Matrix2d d2Jl;
                d2Jl << d2a, d2b, 0.0, d2d;

                // d²Jinv/(dξβ dξγ) = -Jinv*d2Jl*Jinv
                //   + Jinv*dJl[β]*Jinv*dJl[γ]*Jinv
                //   + Jinv*dJl[γ]*Jinv*dJl[β]*Jinv
                Eigen::Matrix2d d2Ji = -Ji * d2Jl * Ji
                    + Ji * dJl[beta] * Ji * dJl[gamma] * Ji
                    + Ji * dJl[gamma] * Ji * dJl[beta] * Ji;

                d2Jinv[beta][gamma](q, 0) = d2Ji(0, 0);
                d2Jinv[beta][gamma](q, 1) = d2Ji(0, 1);
                d2Jinv[beta][gamma](q, 2) = d2Ji(1, 0);
                d2Jinv[beta][gamma](q, 3) = d2Ji(1, 1);
            }
        }
    }
    
    // Fill symmetric d2Jinv[1][0] = d2Jinv[0][1]
    d2Jinv[1][0] = d2Jinv[0][1];

    // ── 3rd order: d³N/(dx_i dx_j dx_k) ──
    // Differentiate 2nd order expression w.r.t. x_k:
    // = Jinv_{γk} ∂/∂ξγ [2nd order expression]
    // 
    // Full expansion:
    // T1: Jinv_{αi} Jinv_{βj} Jinv_{γk} d³N/(dξα dξβ dξγ)
    // T2: [dJinv_{αi}/dξγ Jinv_{βj} Jinv_{γk}
    //     + Jinv_{αi} dJinv_{βj}/dξγ Jinv_{γk}] d²N/(dξα dξβ)    (3 cyclic)
    // T3: Σ_β Jinv_{βk} Σ_γ [Jinv_{γj} d²Jinv_{αi}/(dξβ dξγ)
    //     + dJinv_{γj}/dξβ dJinv_{αi}/dξγ] dN/dξα
    //     + existing Γ terms...
    // 
    // Cleanly: define Γ^α_{ij} as before, then:
    // d³N/(dxi dxj dxk) = Jinv_{αi} Jinv_{βj} Jinv_{γk} d³N/(dξα dξβ dξγ)
    //   + [Γ^α_{ij} Jinv_{βk} + Γ^α_{ik} Jinv_{βj} + Γ^α_{jk} Jinv_{βi}] d²N/(dξα dξβ)
    //   + Γ2^α_{ijk} dN/dξα
    // 
    // where Γ2^α_{ijk} = Σ_γ Jinv_{γk} (dΓ^α_{ij}/dξγ)

    // Pre-compute Γ^α_{ij} at each q for all (i,j)
    // Gamma[i][j][q][α]
    auto compute_Gamma = [&](Eigen::Index q, int i, int j, double& Gu, double& Gv) {
        const Eigen::MatrixXd* dJi[2] = {&dJinv_0, &dJinv_1};
        Gu = 0.0; Gv = 0.0;
        for (int beta = 0; beta < 2; ++beta) {
            double Jibj = Jinv(q, beta * 2 + j);
            Gu += Jibj * (*dJi[beta])(q, 0 * 2 + i);
            Gv += Jibj * (*dJi[beta])(q, 1 * 2 + i);
        }
    };

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = i; j < 2; ++j) {
            for (std::size_t k = j; k < 2; ++k) {
                Eigen::MatrixXd d3N(Q, K);
                for (Eigen::Index q = 0; q < Q; ++q) {
                    double Jpi[2] = {Jinv(q, 0 * 2 + i), Jinv(q, 1 * 2 + i)};
                    double Jpj[2] = {Jinv(q, 0 * 2 + j), Jinv(q, 1 * 2 + j)};
                    double Jpk[2] = {Jinv(q, 0 * 2 + k), Jinv(q, 1 * 2 + k)};

                    // T1: triple Jinv contraction with 3rd parametric derivs
                    Eigen::RowVectorXd t1 = Eigen::RowVectorXd::Zero(K);
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b)
                            for (int c = 0; c < 2; ++c) {
                                int nu = (1-a)+(1-b)+(1-c), nv = a+b+c;
                                t1 += Jpi[a]*Jpj[b]*Jpk[c] * R(nu, nv).row(q);
                            }

                    // T2: Γ corrections on 2nd derivs
                    double Gu_ij, Gv_ij, Gu_ik, Gv_ik, Gu_jk, Gv_jk;
                    compute_Gamma(q, i, j, Gu_ij, Gv_ij);
                    compute_Gamma(q, i, k, Gu_ik, Gv_ik);
                    compute_Gamma(q, j, k, Gu_jk, Gv_jk);

                    Eigen::RowVectorXd t2 = Eigen::RowVectorXd::Zero(K);
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b) {
                            int nu = (1-a)+(1-b), nv = a+b;
                            double G_ij_a = (a==0)?Gu_ij:Gv_ij;
                            double G_ik_a = (a==0)?Gu_ik:Gv_ik;
                            double G_jk_a = (a==0)?Gu_jk:Gv_jk;
                            t2 += (G_ij_a*Jpk[b] + G_ik_a*Jpj[b] + G_jk_a*Jpi[b])
                                  * R(nu, nv).row(q);
                        }

                    // T3: Γ2 correction on 1st derivs
                    // Γ2^α_{ijk} = Σ_γ Jinv_{γk} dΓ^α_{ij}/dξγ
                    // dΓ^α_{ij}/dξγ = Σ_β [dJinv_{βj}/dξγ * dJinv_{αi}/dξβ
                    //                    + Jinv_{βj} * d²Jinv_{αi}/(dξβ dξγ)]
                    const Eigen::MatrixXd* dJi_arr[2] = {&dJinv_0, &dJinv_1};
                    double Gamma2[2] = {0.0, 0.0}; // for α=0,1
                    for (int gamma = 0; gamma < 2; ++gamma) {
                        double Jgk = Jinv(q, gamma * 2 + k);
                        for (int alpha = 0; alpha < 2; ++alpha) {
                            double dGamma = 0.0;
                            for (int beta = 0; beta < 2; ++beta) {
                                double dJibj_dgamma = (*dJi_arr[gamma])(q, beta*2+j);
                                double dJiai_dbeta =  (*dJi_arr[beta])(q, alpha*2+i);
                                double Jibj = Jinv(q, beta*2+j);
                                double d2Jiai = d2Jinv[beta][gamma](q, alpha*2+i);
                                dGamma += dJibj_dgamma * dJiai_dbeta + Jibj * d2Jiai;
                            }
                            Gamma2[alpha] += Jgk * dGamma;
                        }
                    }

                    Eigen::RowVectorXd t3 = Gamma2[0] * R(1, 0).row(q)
                                           + Gamma2[1] * R(0, 1).row(q);

                    d3N.row(q) = t1 + t2 + t3;
                }

                std::string key = "dNd";
                key += coord_label(i); key += 'd';
                key += coord_label(j); key += 'd';
                key += coord_label(k);
                result[key] = d3N;
            }
        }
    }

    return result;
}

// ─── jacobian ───────────────────────────────────────────────────────────────

std::vector<Eigen::MatrixXd> SurfacePatch::jacobian(const Eigen::MatrixXd& params) const
{
    auto derivs = eval_parametric_derivs_(params, 1);
    const auto& dSdu = derivs[1][0];
    const auto& dSdv = derivs[0][1];

    const Eigen::Index Q = dSdu.rows();
    const Eigen::Index d = static_cast<Eigen::Index>(gdim_);

    std::vector<Eigen::MatrixXd> jacobians(Q);
    for (Eigen::Index q = 0; q < Q; ++q) {
        Eigen::MatrixXd J(d, 2);
        J.col(0) = dSdu.row(q).transpose();
        J.col(1) = dSdv.row(q).transpose();
        jacobians[q] = J;
    }

    return jacobians;
}

// ─── jacobian_det ───────────────────────────────────────────────────────────

Eigen::VectorXd SurfacePatch::jacobian_det(const Eigen::MatrixXd& params) const
{
    auto jacs = jacobian(params);
    Eigen::VectorXd dets(jacs.size());
    for (std::size_t q = 0; q < jacs.size(); ++q) {
        Eigen::Matrix2d JtJ = jacs[q].transpose() * jacs[q];
        dets(q) = std::sqrt(JtJ.determinant());
    }
    return dets;
}

} // namespace pyck
