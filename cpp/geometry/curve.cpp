#include "curve.hpp"

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

// ─── factory methods ────────────────────────────────────────────────────────

CurvePatch CurvePatch::line_segment(BSpline* basis, double L)
{
    auto xi = greville_abscissae(*basis);
    std::size_t n = basis->num_basis();

    Eigen::MatrixXd P(n, 3);
    for (std::size_t i = 0; i < n; ++i)
        P.row(i) << L * xi[i], 0.0, 0.0;

    return CurvePatch(3, basis, P);
}

// ─── eval ───────────────────────────────────────────────────────────────────

Eigen::MatrixXd CurvePatch::eval(const Eigen::MatrixXd& params) const
{
    return tensor_product_.eval(params) * control_points_;
}

// ─── private parametric derivative helper ───────────────────────────────────

std::vector<Eigen::MatrixXd>
CurvePatch::eval_parametric_derivs_(const Eigen::MatrixXd& params,
                                    std::size_t order) const
{
    std::vector<Eigen::MatrixXd> result(order + 1);

    if (order == 0) {
        result[0] = eval(params);
    } else {
        auto derivs = tensor_product_.eval_derivs(params, {order});
        for (std::size_t k = 0; k <= order; ++k)
            result[k] = derivs[k] * control_points_;
    }
    return result;
}

// ─── eval_physical_derivs (arc-length derivatives) ──────────────────────────

std::unordered_map<std::string, Eigen::MatrixXd>
CurvePatch::eval_physical_derivs(const Eigen::MatrixXd& params,
                                 std::size_t order) const
{
    order = std::max<std::size_t>(1, std::min<std::size_t>(order, 3));

    const Eigen::Index Q = params.rows();
    const std::size_t K  = tensor_product_.num_basis();

    // ── Parametric basis function derivatives ──
    auto raw = tensor_product_.eval_derivs(params, {order});
    // raw[k] = d^k R / du^k,  shape (Q × K)

    // ── Geometry covariant base vector ──
    Eigen::MatrixXd g1 = raw[1] * control_points_;  // dC/du  (Q × gdim)

    // ── J_local = a = ||dC/du|| ──
    Eigen::VectorXd a_vec(Q);
    Eigen::VectorXd a_inv(Q);
    for (Eigen::Index q = 0; q < Q; ++q) {
        double a = g1.row(q).norm();
        a_vec(q) = a;
        a_inv(q) = 1.0 / a;
    }

    // ── Result map ──
    std::unordered_map<std::string, Eigen::MatrixXd> result;
    result["N"] = raw[0];

    // ── 1st order: dN/dx = (1/a) * dN/du ──
    Eigen::MatrixXd dNdx(Q, K);
    for (Eigen::Index q = 0; q < Q; ++q) {
        dNdx.row(q) = a_inv(q) * raw[1].row(q);
    }
    result["dNdx"] = dNdx;

    if (order < 2) return result;

    // ── Geometry 2nd derivative ──
    Eigen::MatrixXd g1_1 = raw[2] * control_points_;  // d²C/du²  (Q × gdim)

    // ── da/du = (g1 · dg1/du) / a ──
    Eigen::VectorXd da_vec(Q);
    for (Eigen::Index q = 0; q < Q; ++q) {
        da_vec(q) = g1.row(q).dot(g1_1.row(q)) / a_vec(q);
    }

    // ── 2nd order: d²N/dx² = (1/a²) d²N/du² - (da/du / a³) dN/du ──
    // Using: d/dx[dN/dx] = (1/a) d/du[(1/a) dN/du]
    //                    = (1/a) [-da/du / a² * dN/du + (1/a) d²N/du²]
    //                    = (1/a²) d²N/du² - (da/du / a³) dN/du
    Eigen::MatrixXd d2Ndx2(Q, K);
    for (Eigen::Index q = 0; q < Q; ++q) {
        double ai = a_inv(q);
        double ai2 = ai * ai;
        double ai3 = ai2 * ai;
        d2Ndx2.row(q) = ai2 * raw[2].row(q) - (da_vec(q) * ai3) * raw[1].row(q);
    }
    result["dNdxdx"] = d2Ndx2;

    if (order < 3) return result;

    // ── Geometry 3rd derivative ──
    Eigen::MatrixXd g1_11 = raw[3] * control_points_;  // d³C/du³  (Q × gdim)

    // ── d²a/du² = [ dg1/du · dg1/du + g1 · d²g1/du² - (da/du)² ] / a ──
    Eigen::VectorXd d2a_vec(Q);
    for (Eigen::Index q = 0; q < Q; ++q) {
        double a = a_vec(q);
        double da = da_vec(q);
        double num = g1_1.row(q).dot(g1_1.row(q)) + g1.row(q).dot(g1_11.row(q)) - da * da;
        d2a_vec(q) = num / a;
    }

    // ── 3rd order: d³N/dx³ ──
    // Let f = 1/a (Jinv), so dN/dx = f * dN/du
    // d²N/dx² = f * d/du[f * dN/du] = f * [f' dN/du + f d²N/du²]
    //         = f f' dN/du + f² d²N/du²
    // d³N/dx³ = f * d/du[f f' dN/du + f² d²N/du²]
    //         = f * [f'' dN/du + f' d²N/du² ... + ...]
    //
    // More systematically, let Jinv = 1/a, dJinv/du = -a'/a², d²Jinv/du² = (2a'² - a a'')/a³
    //
    // For 1D: d^n N/dx^n is computed recursively.
    // Let φ₁ = dN/dx = Jinv * dN/du
    // Let φ₂ = d²N/dx² = Jinv * d/du[φ₁] = Jinv * [dJinv/du * dN/du + Jinv * d²N/du²]
    // Let φ₃ = d³N/dx³ = Jinv * d/du[φ₂]
    //
    // φ₂ = Jinv * (Jinv' * R₁ + Jinv * R₂)    where Rₖ = d^k N/du^k
    // dφ₂/du = Jinv' * (Jinv' * R₁ + Jinv * R₂)
    //        + Jinv * (Jinv'' * R₁ + Jinv' * R₂ + Jinv' * R₂ + Jinv * R₃)
    //        = Jinv' * Jinv' * R₁ + 2 Jinv * Jinv' * R₂ 
    //          + Jinv * Jinv'' * R₁ + Jinv * Jinv * R₃
    //          + Jinv' * Jinv * R₂
    // φ₃ = Jinv * dφ₂/du
    //    = Jinv * [Jinv'² R₁ + 3 Jinv Jinv' R₂ + Jinv Jinv'' R₁ + Jinv² R₃]
    //
    // With Jinv = 1/a, Jinv' = -a'/a², Jinv'' = (2a'² - a*a'')/a³

    Eigen::MatrixXd d3Ndx3(Q, K);
    for (Eigen::Index q = 0; q < Q; ++q) {
        double a = a_vec(q);
        double da = da_vec(q);
        double d2a = d2a_vec(q);

        double f  = 1.0 / a;
        double fp = -da / (a * a);
        double fpp = (2.0 * da * da - a * d2a) / (a * a * a);

        // φ₃ = f * [fp² R₁ + 3 f fp R₂ + f fpp R₁ + f² R₃]
        //     = f * [(fp² + f fpp) R₁ + 3 f fp R₂ + f² R₃]
        double c1 = f * (fp * fp + f * fpp);
        double c2 = f * 3.0 * f * fp;
        double c3 = f * f * f;

        d3Ndx3.row(q) = c1 * raw[1].row(q) + c2 * raw[2].row(q) + c3 * raw[3].row(q);
    }
    result["dNdxdxdx"] = d3Ndx3;

    return result;
}

// ─── jacobian ───────────────────────────────────────────────────────────────

std::vector<Eigen::MatrixXd> CurvePatch::jacobian(const Eigen::MatrixXd& params) const
{
    auto derivs = eval_parametric_derivs_(params, 1);
    const auto& dCdu = derivs[1];

    const Eigen::Index Q = dCdu.rows();
    const Eigen::Index d = static_cast<Eigen::Index>(gdim_);

    std::vector<Eigen::MatrixXd> jacobians(Q);
    for (Eigen::Index q = 0; q < Q; ++q) {
        Eigen::MatrixXd J(d, 1);
        J.col(0) = dCdu.row(q).transpose();
        jacobians[q] = J;
    }

    return jacobians;
}

// ─── jacobian_det ───────────────────────────────────────────────────────────

Eigen::VectorXd CurvePatch::jacobian_det(const Eigen::MatrixXd& params) const
{
    auto jacs = jacobian(params);
    Eigen::VectorXd dets(jacs.size());
    for (std::size_t q = 0; q < jacs.size(); ++q) {
        dets(q) = jacs[q].col(0).norm();
    }
    return dets;
}

} // namespace pyck
