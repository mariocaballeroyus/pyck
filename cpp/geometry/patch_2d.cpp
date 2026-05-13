#include "patch.hpp"
#include "bspline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../quadrature/quadrature.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u, Ptr<const Basis<T>> basis_v, const ColMatrix<T, 3>& control_pts) requires(d == 2)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u), std::move(basis_v)),
      dof_mapper_({tensor_product_.basis(0).num_basis(), tensor_product_.basis(1).num_basis()},
                  {tensor_product_.basis(0).degree(),    tensor_product_.basis(1).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("Patch<T, 2>: Control points must be embedded in 3D space.");
    }
    const Index expected_n = this->tensor_product_.basis(0).num_basis() * this->tensor_product_.basis(1).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());
    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch<T, 2>: Dimension mismatch.");
    }

    // Initialize Greville points and spans
    Vector<T> gu = this->tensor_product_.basis(0).greville_abscissae();
    Vector<T> gv = this->tensor_product_.basis(1).greville_abscissae();
    Index nu = gu.size();
    Index nv = gv.size();
    
    greville_points_.resize(nu * nv, 2);
    greville_spans_.resize(nu * nv);
    
    auto intervals = this->tensor_product_.num_intervals();
    for (Index i = 0; i < nu; ++i) {
        Index su = this->tensor_product_.basis(0).find_span(gu[i]);
        for (Index j = 0; j < nv; ++j) {
            Index flat = i * nv + j;
            greville_points_(flat, 0) = gu[i];
            greville_points_(flat, 1) = gv[j];
            Index sv = this->tensor_product_.basis(1).find_span(gv[j]);
            greville_spans_[flat] = su * intervals[1] + sv;
        }
    }
}

#if 0  // Removed: eval_shape_functions superseded by eval_basis + eval_local_frame.
template <std::floating_point T, std::size_t d>
std::pair<std::vector<Matrix<T>>, Vector<T>>
Patch<T, d>::eval_shape_functions(const ColMatrix<T, d>& eval_coords, Index span_idx, std::size_t derivs_order) const requires(d == 2)
{
    auto spans = this->decode_span(span_idx);
    Index order = std::max(Index(1), std::min(static_cast<Index>(derivs_order), Index(3)));
    const Index Q = eval_coords.rows();

    // Parametric (u, v) basis derivatives. For order=k the tensor product
    // returns (k+1)^2 matrices laid out as idx = du * (k+1) + dv.
    auto basis_derivs = this->tensor_product().eval_on_span(eval_coords, spans, order);
    const Index K = basis_derivs[0].cols();
    const Index S = order + 1;
    auto act_pts = this->active_control_pts(spans);

    // Geometric tangents a_α = ∂x/∂ξ^α (Q × 3, z-component zero for plates).
    ColMatrix<T, 3> a_1 = basis_derivs[1 * S + 0] * act_pts;
    ColMatrix<T, 3> a_2 = basis_derivs[0 * S + 1] * act_pts;

    const size_t num_results = (order == 1) ? 3 : (order == 2 ? 6 : 10);
    std::vector<Matrix<T>> result(num_results);
    Vector<T> jacobian(Q);

    result[0] = basis_derivs[0];
    result[1] = Matrix<T>(Q, K);  // N,x  (Cartesian)
    result[2] = Matrix<T>(Q, K);  // N,y  (Cartesian)

    const Matrix<T>& N_u = basis_derivs[1 * S + 0];
    const Matrix<T>& N_v = basis_derivs[0 * S + 1];

    // PYCK_PATCH2D_FIX_MARKER_v1
    // First-order chain rule: [N,x; N,y] = J^{-T} [N,u; N,v], where
    //   J = [x_u, x_v; y_u, y_v],   J^{-1} = (1/det J)[y_v, -x_v; -y_u, x_u].
    // The integration weight is the surface area element sqrt(det g) where
    // g is the first fundamental form; for flat plates this equals |det J|.
    for (Index q = 0; q < Q; ++q)
    {
        const T xu = a_1(q, 0), yu = a_1(q, 1), zu = a_1(q, 2);
        const T xv = a_2(q, 0), yv = a_2(q, 1), zv = a_2(q, 2);
        const T g11 = xu * xu + yu * yu + zu * zu;
        const T g12 = xu * xv + yu * yv + zu * zv;
        const T g22 = xv * xv + yv * yv + zv * zv;
        jacobian(q) = std::sqrt(g11 * g22 - g12 * g12);
        const T detJ = xu * yv - xv * yu;
        const T inv_detJ = T(1) / detJ;
        result[1].row(q) = inv_detJ * (yv * N_u.row(q) - yu * N_v.row(q));
        result[2].row(q) = inv_detJ * (xu * N_v.row(q) - xv * N_u.row(q));
    }

    if (order >= 2)
    {
        ColMatrix<T, 3> a_11 = basis_derivs[2 * S + 0] * act_pts;
        ColMatrix<T, 3> a_12 = basis_derivs[1 * S + 1] * act_pts;
        ColMatrix<T, 3> a_22 = basis_derivs[0 * S + 2] * act_pts;

        const Matrix<T>& N_uu = basis_derivs[2 * S + 0];
        const Matrix<T>& N_uv = basis_derivs[1 * S + 1];
        const Matrix<T>& N_vv = basis_derivs[0 * S + 2];

        result[3] = Matrix<T>(Q, K);  // N,xx
        result[4] = Matrix<T>(Q, K);  // N,xy
        result[5] = Matrix<T>(Q, K);  // N,yy

        // Second-order chain rule. With H_2 the 3x3 matrix relating
        // [N,uu; N,uv; N,vv] to [N,xx; N,xy; N,yy] for an affine map, plus
        // the geometry-second-derivative correction picked up by the linear
        // term, solve  H_2 [N,xx; N,xy; N,yy] = [N,uu; N,uv; N,vv] - corr.
        for (Index q = 0; q < Q; ++q)
        {
            const T xu = a_1(q, 0),  yu = a_1(q, 1);
            const T xv = a_2(q, 0),  yv = a_2(q, 1);
            const T xuu = a_11(q, 0), yuu = a_11(q, 1);
            const T xuv = a_12(q, 0), yuv = a_12(q, 1);
            const T xvv = a_22(q, 0), yvv = a_22(q, 1);

            Eigen::Matrix<T, 3, 3> H2;
            H2 << xu * xu,        T(2) * xu * yu,        yu * yu,
                  xu * xv,        xu * yv + xv * yu,     yu * yv,
                  xv * xv,        T(2) * xv * yv,        yv * yv;
            const Eigen::Matrix<T, 3, 3> H2inv = H2.inverse();

            Matrix<T> rhs(3, K);
            rhs.row(0) = N_uu.row(q) - xuu * result[1].row(q) - yuu * result[2].row(q);
            rhs.row(1) = N_uv.row(q) - xuv * result[1].row(q) - yuv * result[2].row(q);
            rhs.row(2) = N_vv.row(q) - xvv * result[1].row(q) - yvv * result[2].row(q);

            const Matrix<T> cart = H2inv * rhs;
            result[3].row(q) = cart.row(0);
            result[4].row(q) = cart.row(1);
            result[5].row(q) = cart.row(2);
        }
    }

    if (order >= 3)
    {
        ColMatrix<T, 3> a_11  = basis_derivs[2 * S + 0] * act_pts;
        ColMatrix<T, 3> a_12  = basis_derivs[1 * S + 1] * act_pts;
        ColMatrix<T, 3> a_22  = basis_derivs[0 * S + 2] * act_pts;
        ColMatrix<T, 3> a_111 = basis_derivs[3 * S + 0] * act_pts;
        ColMatrix<T, 3> a_112 = basis_derivs[2 * S + 1] * act_pts;
        ColMatrix<T, 3> a_122 = basis_derivs[1 * S + 2] * act_pts;
        ColMatrix<T, 3> a_222 = basis_derivs[0 * S + 3] * act_pts;

        const Matrix<T>& N_uuu = basis_derivs[3 * S + 0];
        const Matrix<T>& N_uuv = basis_derivs[2 * S + 1];
        const Matrix<T>& N_uvv = basis_derivs[1 * S + 2];
        const Matrix<T>& N_vvv = basis_derivs[0 * S + 3];

        result[6] = Matrix<T>(Q, K);  // N,xxx
        result[7] = Matrix<T>(Q, K);  // N,xxy
        result[8] = Matrix<T>(Q, K);  // N,xyy
        result[9] = Matrix<T>(Q, K);  // N,yyy

        // Third-order chain rule. The 4x4 matrix is the trinomial expansion
        // of (x_u ∂_x + y_u ∂_y)^a (x_v ∂_x + y_v ∂_y)^b acting on N for
        // (a, b) = (3,0), (2,1), (1,2), (0,3); the correction subtracts
        // contributions from second Cartesian derivatives × second
        // geometric derivatives, and first Cartesian × third geometric.
        for (Index q = 0; q < Q; ++q)
        {
            const T xu = a_1(q, 0),   yu = a_1(q, 1);
            const T xv = a_2(q, 0),   yv = a_2(q, 1);
            const T xuu = a_11(q, 0),  yuu = a_11(q, 1);
            const T xuv = a_12(q, 0),  yuv = a_12(q, 1);
            const T xvv = a_22(q, 0),  yvv = a_22(q, 1);
            const T xuuu = a_111(q, 0), yuuu = a_111(q, 1);
            const T xuuv = a_112(q, 0), yuuv = a_112(q, 1);
            const T xuvv = a_122(q, 0), yuvv = a_122(q, 1);
            const T xvvv = a_222(q, 0), yvvv = a_222(q, 1);

            Eigen::Matrix<T, 4, 4> H3;
            // N_uuu: (3,0)
            H3.row(0) << xu*xu*xu,
                         T(3)*xu*xu*yu,
                         T(3)*xu*yu*yu,
                         yu*yu*yu;
            // N_uuv: (2,1)
            H3.row(1) << xu*xu*xv,
                         xu*xu*yv + T(2)*xu*xv*yu,
                         T(2)*xu*yu*yv + xv*yu*yu,
                         yu*yu*yv;
            // N_uvv: (1,2)
            H3.row(2) << xu*xv*xv,
                         T(2)*xu*xv*yv + xv*xv*yu,
                         xu*yv*yv + T(2)*xv*yu*yv,
                         yu*yv*yv;
            // N_vvv: (0,3)
            H3.row(3) << xv*xv*xv,
                         T(3)*xv*xv*yv,
                         T(3)*xv*yv*yv,
                         yv*yv*yv;
            const Eigen::Matrix<T, 4, 4> H3inv = H3.inverse();

            // Corrections: derived from differentiating the second-order
            // chain rule once more in u or v.
            // N_uuu: 3 (x_u x_uu Nxx + (x_uu y_u + x_u y_uu) Nxy + y_u y_uu Nyy) + x_uuu N_x + y_uuu N_y
            // N_uuv: (2 x_u x_uv + x_uu x_v) Nxx + (2 x_uv y_u + 2 x_u y_uv + x_uu y_v + y_uu x_v) Nxy + (2 y_u y_uv + y_uu y_v) Nyy + x_uuv N_x + y_uuv N_y
            // N_uvv: (2 x_uv x_v + x_u x_vv) Nxx + (2 x_uv y_v + 2 x_v y_uv + x_u y_vv + x_vv y_u) Nxy + (2 y_uv y_v + y_u y_vv) Nyy + x_uvv N_x + y_uvv N_y
            // N_vvv: 3 (x_v x_vv Nxx + (x_vv y_v + x_v y_vv) Nxy + y_v y_vv Nyy) + x_vvv N_x + y_vvv N_y
            Matrix<T> rhs(4, K);

            rhs.row(0) = N_uuu.row(q)
                       - T(3) * (xu * xuu)            * result[3].row(q)
                       - T(3) * (xuu * yu + xu * yuu) * result[4].row(q)
                       - T(3) * (yu * yuu)            * result[5].row(q)
                       - xuuu * result[1].row(q) - yuuu * result[2].row(q);

            rhs.row(1) = N_uuv.row(q)
                       - (T(2)*xu*xuv + xuu*xv)                              * result[3].row(q)
                       - (T(2)*xuv*yu + T(2)*xu*yuv + xuu*yv + yuu*xv)        * result[4].row(q)
                       - (T(2)*yu*yuv + yuu*yv)                              * result[5].row(q)
                       - xuuv * result[1].row(q) - yuuv * result[2].row(q);

            rhs.row(2) = N_uvv.row(q)
                       - (T(2)*xuv*xv + xu*xvv)                              * result[3].row(q)
                       - (T(2)*xuv*yv + T(2)*xv*yuv + xu*yvv + xvv*yu)        * result[4].row(q)
                       - (T(2)*yuv*yv + yu*yvv)                              * result[5].row(q)
                       - xuvv * result[1].row(q) - yuvv * result[2].row(q);

            rhs.row(3) = N_vvv.row(q)
                       - T(3) * (xv * xvv)            * result[3].row(q)
                       - T(3) * (xvv * yv + xv * yvv) * result[4].row(q)
                       - T(3) * (yv * yvv)            * result[5].row(q)
                       - xvvv * result[1].row(q) - yvvv * result[2].row(q);

            const Matrix<T> cart3 = H3inv * rhs;
            result[6].row(q) = cart3.row(0);
            result[7].row(q) = cart3.row(1);
            result[8].row(q) = cart3.row(2);
            result[9].row(q) = cart3.row(3);
        }
    }

    return {std::move(result), std::move(jacobian)};
}
#endif  // end of removed eval_shape_functions(d == 2)


// ============================================================================
// Composable geometric primitives — 2D
// ============================================================================

template <std::floating_point T, std::size_t d>
BasisDerivs<T, d>
Patch<T, d>::eval_basis(const ColMatrix<T, d>& eval_coords,
                        Index span_idx,
                        std::size_t derivs_order) const requires(d == 2)
{
    auto spans = this->decode_span(span_idx);
    auto raw = this->tensor_product().eval_on_span(eval_coords, spans,
                                                   static_cast<Index>(derivs_order));
    BasisDerivs<T, 2> b;
    b.order = static_cast<Index>(derivs_order);
    const Index S = b.order + 1;
    b.N = std::move(raw[0]);
    if (b.order >= 1) {
        b.N_u = std::move(raw[1 * S + 0]);
        b.N_v = std::move(raw[0 * S + 1]);
    }
    if (b.order >= 2) {
        b.N_uu = std::move(raw[2 * S + 0]);
        b.N_uv = std::move(raw[1 * S + 1]);
        b.N_vv = std::move(raw[0 * S + 2]);
    }
    if (b.order >= 3) {
        b.N_uuu = std::move(raw[3 * S + 0]);
        b.N_uuv = std::move(raw[2 * S + 1]);
        b.N_uvv = std::move(raw[1 * S + 2]);
        b.N_vvv = std::move(raw[0 * S + 3]);
    }
    return b;
}

template <std::floating_point T, std::size_t d>
LocalFrame<T, d>
Patch<T, d>::eval_local_frame(const BasisDerivs<T, d>& basis,
                              const ColMatrix<T, 3>& act_pts) const requires(d == 2)
{
    const Index Q = basis.N.rows();

    LocalFrame<T, 2> lf;

    // Tangents a_α (order ≥ 1).
    lf.a1 = basis.N_u * act_pts;
    lf.a2 = basis.N_v * act_pts;

    // Tangent derivatives a_{αβ} (order ≥ 2).
    if (basis.order >= 2) {
        lf.a11 = basis.N_uu * act_pts;
        lf.a12 = basis.N_uv * act_pts;
        lf.a22 = basis.N_vv * act_pts;
    }

    // 2nd tangent derivatives a_{αβγ} (order ≥ 3).
    if (basis.order >= 3) {
        lf.a111 = basis.N_uuu * act_pts;
        lf.a112 = basis.N_uuv * act_pts;
        lf.a122 = basis.N_uvv * act_pts;
        lf.a222 = basis.N_vvv * act_pts;
    }

    lf.g.resize(Q, 3);
    lf.g_inv.resize(Q, 3);
    lf.jac.resize(Q);

    for (Index q = 0; q < Q; ++q) {
        const auto a1_q = lf.a1.row(q);
        const auto a2_q = lf.a2.row(q);
        const T g11 = a1_q.squaredNorm();
        const T g12 = a1_q.dot(a2_q);
        const T g22 = a2_q.squaredNorm();
        const T det_g = g11 * g22 - g12 * g12;
        const T inv_dg = T(1) / det_g;
        lf.g(q, 0) = g11;
        lf.g(q, 1) = g12;
        lf.g(q, 2) = g22;
        lf.g_inv(q, 0) =  g22 * inv_dg;
        lf.g_inv(q, 1) = -g12 * inv_dg;
        lf.g_inv(q, 2) =  g11 * inv_dg;
        lf.jac(q) = std::sqrt(det_g);
    }
    return lf;
}

template <std::floating_point T, std::size_t d>
ChristoffelSymbols<T, d>
Patch<T, d>::eval_christoffel(const LocalFrame<T, d>& local) const requires(d == 2)
{
    const Index Q = local.a1.rows();
    ChristoffelSymbols<T, d> chr;
    chr.G1_11.resize(Q); chr.G1_12.resize(Q); chr.G1_22.resize(Q);
    chr.G2_11.resize(Q); chr.G2_12.resize(Q); chr.G2_22.resize(Q);
    for (Index q = 0; q < Q; ++q) {
        // Dual basis a^δ = g^{δβ} a_β built as a per-qp local — never exposed.
        const auto a1_q = local.a1.row(q);
        const auto a2_q = local.a2.row(q);
        const T gi11 = local.g_inv(q, 0);
        const T gi12 = local.g_inv(q, 1);
        const T gi22 = local.g_inv(q, 2);
        const Eigen::RowVector<T, 3> aup1 = gi11 * a1_q + gi12 * a2_q;
        const Eigen::RowVector<T, 3> aup2 = gi12 * a1_q + gi22 * a2_q;

        const auto a11_q = local.a11.row(q);
        const auto a12_q = local.a12.row(q);
        const auto a22_q = local.a22.row(q);
        chr.G1_11(q) = aup1.dot(a11_q);
        chr.G1_12(q) = aup1.dot(a12_q);
        chr.G1_22(q) = aup1.dot(a22_q);
        chr.G2_11(q) = aup2.dot(a11_q);
        chr.G2_12(q) = aup2.dot(a12_q);
        chr.G2_22(q) = aup2.dot(a22_q);
    }
    return chr;
}

template <std::floating_point T>
LaplaceGradAux<T> compute_laplace_grad_aux(
    const LocalFrame<T, 2>& local,
    const ChristoffelSymbols<T, 2>& chr)
{
    const Index Q = local.a1.rows();

    LaplaceGradAux<T> aux;
    aux.G11_d1.resize(Q); aux.G12_d1.resize(Q); aux.G22_d1.resize(Q);
    aux.G11_d2.resize(Q); aux.G12_d2.resize(Q); aux.G22_d2.resize(Q);
    aux.c1.resize(Q);     aux.c2.resize(Q);
    aux.c1_d1.resize(Q);  aux.c2_d1.resize(Q);
    aux.c1_d2.resize(Q);  aux.c2_d2.resize(Q);

    for (Index q = 0; q < Q; ++q) {
        const T G11 = local.g_inv(q, 0);
        const T G12 = local.g_inv(q, 1);
        const T G22 = local.g_inv(q, 2);
        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);

        const auto a1   = local.a1.row(q);
        const auto a2   = local.a2.row(q);
        const auto a11  = local.a11.row(q);
        const auto a12  = local.a12.row(q);
        const auto a22  = local.a22.row(q);
        const auto a111 = local.a111.row(q);
        const auto a112 = local.a112.row(q);
        const auto a122 = local.a122.row(q);
        const auto a222 = local.a222.row(q);

        // Tangent dot products.
        const T a1_a11  = a1.dot(a11),  a1_a12  = a1.dot(a12),  a1_a22  = a1.dot(a22);
        const T a2_a11  = a2.dot(a11),  a2_a12  = a2.dot(a12),  a2_a22  = a2.dot(a22);
        const T a11_a11 = a11.squaredNorm(), a11_a12 = a11.dot(a12), a11_a22 = a11.dot(a22);
        const T a12_a12 = a12.squaredNorm(), a12_a22 = a12.dot(a22), a22_a22 = a22.squaredNorm();
        const T a1_a111 = a1.dot(a111), a1_a112 = a1.dot(a112);
        const T a1_a122 = a1.dot(a122), a1_a222 = a1.dot(a222);
        const T a2_a111 = a2.dot(a111), a2_a112 = a2.dot(a112);
        const T a2_a122 = a2.dot(a122), a2_a222 = a2.dot(a222);

        // Metric partials g_{μν,γ} = a_{μγ}·a_ν + a_μ·a_{νγ}.
        const T g11_d1 = T(2) * a1_a11;
        const T g11_d2 = T(2) * a1_a12;
        const T g12_d1 = a2_a11 + a1_a12;
        const T g12_d2 = a2_a12 + a1_a22;
        const T g22_d1 = T(2) * a2_a12;
        const T g22_d2 = T(2) * a2_a22;

        // Inverse-metric partials (g^{βγ})_{,α} = -g^{βμ} g^{γν} g_{μν,α}.
        const T G11_d1 = -(G11*G11*g11_d1 + T(2)*G11*G12*g12_d1 + G12*G12*g22_d1);
        const T G11_d2 = -(G11*G11*g11_d2 + T(2)*G11*G12*g12_d2 + G12*G12*g22_d2);
        const T G12_d1 = -(G11*G12*g11_d1 + (G11*G22 + G12*G12)*g12_d1 + G12*G22*g22_d1);
        const T G12_d2 = -(G11*G12*g11_d2 + (G11*G22 + G12*G12)*g12_d2 + G12*G22*g22_d2);
        const T G22_d1 = -(G12*G12*g11_d1 + T(2)*G12*G22*g12_d1 + G22*G22*g22_d1);
        const T G22_d2 = -(G12*G12*g11_d2 + T(2)*G12*G22*g12_d2 + G22*G22*g22_d2);

        // Γ^δ_{βγ,α} = (g^{δε})_{,α}(a_ε·a_{βγ}) + g^{δε}(a_{εα}·a_{βγ} + a_ε·a_{βγα}).
        const T G1_11_d1 = G11_d1*a1_a11 + G12_d1*a2_a11
                         + G11*(a11_a11 + a1_a111) + G12*(a11_a12 + a2_a111);
        const T G1_12_d1 = G11_d1*a1_a12 + G12_d1*a2_a12
                         + G11*(a11_a12 + a1_a112) + G12*(a12_a12 + a2_a112);
        const T G1_22_d1 = G11_d1*a1_a22 + G12_d1*a2_a22
                         + G11*(a11_a22 + a1_a122) + G12*(a12_a22 + a2_a122);
        const T G2_11_d1 = G12_d1*a1_a11 + G22_d1*a2_a11
                         + G12*(a11_a11 + a1_a111) + G22*(a11_a12 + a2_a111);
        const T G2_12_d1 = G12_d1*a1_a12 + G22_d1*a2_a12
                         + G12*(a11_a12 + a1_a112) + G22*(a12_a12 + a2_a112);
        const T G2_22_d1 = G12_d1*a1_a22 + G22_d1*a2_a22
                         + G12*(a11_a22 + a1_a122) + G22*(a12_a22 + a2_a122);
        const T G1_11_d2 = G11_d2*a1_a11 + G12_d2*a2_a11
                         + G11*(a11_a12 + a1_a112) + G12*(a11_a22 + a2_a112);
        const T G1_12_d2 = G11_d2*a1_a12 + G12_d2*a2_a12
                         + G11*(a12_a12 + a1_a122) + G12*(a12_a22 + a2_a122);
        const T G1_22_d2 = G11_d2*a1_a22 + G12_d2*a2_a22
                         + G11*(a12_a22 + a1_a222) + G12*(a22_a22 + a2_a222);
        const T G2_11_d2 = G12_d2*a1_a11 + G22_d2*a2_a11
                         + G12*(a11_a12 + a1_a112) + G22*(a11_a22 + a2_a112);
        const T G2_12_d2 = G12_d2*a1_a12 + G22_d2*a2_a12
                         + G12*(a12_a12 + a1_a122) + G22*(a12_a22 + a2_a122);
        const T G2_22_d2 = G12_d2*a1_a22 + G22_d2*a2_a22
                         + G12*(a12_a22 + a1_a222) + G22*(a22_a22 + a2_a222);

        // c^δ and (c^δ)_{,α}.
        const T c1 = G11*Gam1_11 + T(2)*G12*Gam1_12 + G22*Gam1_22;
        const T c2 = G11*Gam2_11 + T(2)*G12*Gam2_12 + G22*Gam2_22;
        const T c1_d1 = G11_d1*Gam1_11 + T(2)*G12_d1*Gam1_12 + G22_d1*Gam1_22
                      + G11*G1_11_d1 + T(2)*G12*G1_12_d1 + G22*G1_22_d1;
        const T c2_d1 = G11_d1*Gam2_11 + T(2)*G12_d1*Gam2_12 + G22_d1*Gam2_22
                      + G11*G2_11_d1 + T(2)*G12*G2_12_d1 + G22*G2_22_d1;
        const T c1_d2 = G11_d2*Gam1_11 + T(2)*G12_d2*Gam1_12 + G22_d2*Gam1_22
                      + G11*G1_11_d2 + T(2)*G12*G1_12_d2 + G22*G1_22_d2;
        const T c2_d2 = G11_d2*Gam2_11 + T(2)*G12_d2*Gam2_12 + G22_d2*Gam2_22
                      + G11*G2_11_d2 + T(2)*G12*G2_12_d2 + G22*G2_22_d2;

        aux.G11_d1(q) = G11_d1; aux.G12_d1(q) = G12_d1; aux.G22_d1(q) = G22_d1;
        aux.G11_d2(q) = G11_d2; aux.G12_d2(q) = G12_d2; aux.G22_d2(q) = G22_d2;
        aux.c1(q)     = c1;     aux.c2(q)     = c2;
        aux.c1_d1(q)  = c1_d1;  aux.c2_d1(q)  = c2_d1;
        aux.c1_d2(q)  = c1_d2;  aux.c2_d2(q)  = c2_d2;
    }

    return aux;
}

template LaplaceGradAux<double> compute_laplace_grad_aux<double>(
    const LocalFrame<double, 2>&, const ChristoffelSymbols<double, 2>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template LaplaceGradAux<float> compute_laplace_grad_aux<float>(
    const LocalFrame<float, 2>&, const ChristoffelSymbols<float, 2>&);
#endif

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::eval_normal(const LocalFrame<T, 2>& local) const requires(d == 2)
{
    const Index Q = local.a1.rows();
    ColMatrix<T, 3> a_3(Q, 3);
    for (Index q = 0; q < Q; ++q) {
        Eigen::Matrix<T, 3, 1> n =
            local.a1.row(q).transpose().cross(local.a2.row(q).transpose());
        const T J = local.jac(q);
        if (J > T(1e-14)) n /= J;
        else n = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
        a_3.row(q) = n.transpose();
    }
    return a_3;
}

template <std::floating_point T, std::size_t d>
std::pair<ColMatrix<T, 3>, ColMatrix<T, 3>>
Patch<T, d>::eval_normal_deriv(const LocalFrame<T, 2>& local,
                               const ColMatrix<T, 3>& a_3) const requires(d == 2)
{
    const Index Q = local.a1.rows();
    ColMatrix<T, 3> a_3_1(Q, 3);
    ColMatrix<T, 3> a_3_2(Q, 3);
    for (Index q = 0; q < Q; ++q) {
        const auto a1_q  = local.a1.row(q);
        const auto a2_q  = local.a2.row(q);
        const auto a3_q  = a_3.row(q);
        const auto a11_q = local.a11.row(q);
        const auto a12_q = local.a12.row(q);
        const auto a22_q = local.a22.row(q);
        const T gi11 = local.g_inv(q, 0);
        const T gi12 = local.g_inv(q, 1);
        const T gi22 = local.g_inv(q, 2);

        // Second fundamental form b_{αβ} = a_{αβ}·a_3 — local, not exposed.
        const T b11 = a11_q.dot(a3_q);
        const T b12 = a12_q.dot(a3_q);
        const T b22 = a22_q.dot(a3_q);
        const T bup11 = gi11 * b11 + gi12 * b12;
        const T bup12 = gi11 * b12 + gi12 * b22;
        const T bup21 = gi12 * b11 + gi22 * b12;
        const T bup22 = gi12 * b12 + gi22 * b22;
        a_3_1.row(q) = -(bup11 * a1_q + bup21 * a2_q);
        a_3_2.row(q) = -(bup12 * a1_q + bup22 * a2_q);
    }
    return {std::move(a_3_1), std::move(a_3_2)};
}


template class Patch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 2>;
#endif

} // namespace pyck
