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

template <std::floating_point T, std::size_t d>
std::pair<std::vector<Matrix<T>>, Vector<T>> 
Patch<T, d>::eval_shape_functions_at_greville(Index dof_index, std::size_t order_in) const 
{
    ColMatrix<T, d> pt = greville_points_.row(dof_index);
    Index span = greville_spans_[dof_index];
    return eval_shape_functions(pt, span, order_in);
}

template <std::floating_point T, std::size_t d>
std::pair<std::vector<Matrix<T>>, Vector<T>>
Patch<T, d>::eval_shape_functions(const ColMatrix<T, d>& points, Index span, std::size_t order_in) const requires(d == 2)
{
    auto spans = this->decode_span(span);
    Index order = std::max(Index(1), std::min(static_cast<Index>(order_in), Index(3)));
    const Index Q = points.rows();

    // Parametric (u, v) basis derivatives. For order=k the tensor product
    // returns (k+1)^2 matrices laid out as idx = du * (k+1) + dv.
    auto basis_derivs = this->tensor_product().eval_on_span(points, spans, order);
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


template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::eval_geometry(const ColMatrix<T, d>& points, Index span) const requires(d == 2)
{
    auto spans = this->decode_span(span);
    auto basis_fns = this->tensor_product().eval_on_span(points, spans, 0);
    auto act_pts = this->active_control_pts(spans);
    return basis_fns[0] * act_pts;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::eval_physical_points(const QuadratureRule<T, d>& quadrature) const requires(d == 2)
{
    auto intervals = this->tensor_product().num_intervals();
    const Index total_elements = intervals[0] * intervals[1];
    const Index Q = static_cast<Index>(quadrature.num_points());

    ColMatrix<T, 3> result(total_elements * Q, 3);
    Index out = 0;

    for (Index elem_idx = 0; elem_idx < total_elements; ++elem_idx)
    {
        auto spans = this->decode_span(elem_idx);
        std::array<T, 2> lo, hi;
        bool zero_volume = false;
        for (std::size_t i = 0; i < 2; ++i)
        {
            auto [l, h] = this->basis(i).knot_vector().span_bounds(spans[i]);
            lo[i] = l;
            hi[i] = h;
            if (std::abs(h - l) < T(1e-14))
            {
                zero_volume = true;
                break;
            }
        }
        if (zero_volume) continue;

        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        auto geom = this->eval_geometry(mapped_pts, elem_idx);
        result.block(out, 0, Q, 3) = geom;
        out += Q;
    }

    result.conservativeResize(out, Eigen::NoChange);
    return result;
}

template <std::floating_point T, std::size_t d>
std::array<ColMatrix<T, 3>, d> Patch<T, d>::eval_tangent(
    const std::vector<Matrix<T>>& N,
    const ColMatrix<T, 3>& act_pts) const requires(d == 2)
{
    std::array<ColMatrix<T, 3>, 2> tangents;
    if (N.size() < 3) throw std::runtime_error("eval_tangent: N must contain at least 3 entries (values and derivatives)");
    tangents[0] = N[1] * act_pts;  // ∂x/∂ξ⁰
    tangents[1] = N[2] * act_pts;  // ∂x/∂ξ¹
    return tangents;
}

template <std::floating_point T, std::size_t d>
std::tuple<ColMatrix<T, 3>, ColMatrix<T, 3>, ColMatrix<T, 3>, Vector<T>>
Patch<T, d>::eval_local_frame(const ColMatrix<T, d>& points,
                              Index span) const requires(d == 2)
{
    auto spans = this->decode_span(span);
    auto basis_derivs = this->tensor_product().eval_on_span(points, spans, Index(1));
    auto act_pts = this->active_control_pts(spans);

    const Index Q = points.rows();
    const Index S = 2;  // (order + 1) for order = 1

    ColMatrix<T, 3> a1 = basis_derivs[1 * S + 0] * act_pts;  // ∂x/∂ξ⁰
    ColMatrix<T, 3> a2 = basis_derivs[0 * S + 1] * act_pts;  // ∂x/∂ξ¹
    ColMatrix<T, 3> a3(Q, 3);
    Vector<T> jac(Q);

    for (Index q = 0; q < Q; ++q) {
        Eigen::Matrix<T, 3, 1> a1_q = a1.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a2_q = a2.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a3_q = a1_q.cross(a2_q);
        const T n = a3_q.norm();
        jac(q) = n;
        if (n > T(1e-14)) a3_q /= n;
        else a3_q = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
        a3.row(q) = a3_q.transpose();
    }
    return {std::move(a1), std::move(a2), std::move(a3), std::move(jac)};
}

template <std::floating_point T, std::size_t d>
SurfaceKinematics<T>
Patch<T, d>::eval_surface_kinematics(const ColMatrix<T, d>& points,
                                     Index span,
                                     std::size_t order_in) const requires(d == 2)
{
    const Index order = std::max(Index(1), std::min(static_cast<Index>(order_in), Index(3)));
    auto spans   = this->decode_span(span);
    auto act_pts = this->active_control_pts(spans);

    SurfaceKinematics<T> sk;
    sk.basis_derivs = this->tensor_product().eval_on_span(points, spans, order);
    const Index Q = points.rows();
    const Index S = order + 1;

    sk.a1 = sk.basis_derivs[1 * S + 0] * act_pts;
    sk.a2 = sk.basis_derivs[0 * S + 1] * act_pts;

    sk.a3.resize(Q, 3);
    sk.aup1.resize(Q, 3);
    sk.aup2.resize(Q, 3);
    sk.g_inv.resize(Q, 3);
    sk.jac.resize(Q);

    if (order >= 2) {
        sk.a11 = sk.basis_derivs[2 * S + 0] * act_pts;
        sk.a12 = sk.basis_derivs[1 * S + 1] * act_pts;
        sk.a22 = sk.basis_derivs[0 * S + 2] * act_pts;
        sk.b.resize(Q, 3);
        sk.christoffel.resize(Q, 6);
        sk.a3_1.resize(Q, 3);
        sk.a3_2.resize(Q, 3);
    }
    if (order >= 3) {
        sk.a111 = sk.basis_derivs[3 * S + 0] * act_pts;
        sk.a112 = sk.basis_derivs[2 * S + 1] * act_pts;
        sk.a122 = sk.basis_derivs[1 * S + 2] * act_pts;
        sk.a222 = sk.basis_derivs[0 * S + 3] * act_pts;
    }

    for (Index q = 0; q < Q; ++q) {
        const auto a1_q = sk.a1.row(q);
        const auto a2_q = sk.a2.row(q);

        // Director a3 = a1 × a2, with jacobian = ||a1 × a2||.
        Eigen::Matrix<T, 3, 1> a3_q = a1_q.transpose().cross(a2_q.transpose());
        const T n = a3_q.norm();
        sk.jac(q) = n;
        if (n > T(1e-14)) a3_q /= n;
        else a3_q = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
        sk.a3.row(q) = a3_q.transpose();

        // Metric.
        const T g11 = a1_q.squaredNorm();
        const T g12 = a1_q.dot(a2_q);
        const T g22 = a2_q.squaredNorm();
        const T det_g  = g11 * g22 - g12 * g12;
        const T inv_dg = T(1) / det_g;
        const T gi11 =  g22 * inv_dg;
        const T gi12 = -g12 * inv_dg;
        const T gi22 =  g11 * inv_dg;
        sk.g_inv(q, 0) = gi11;
        sk.g_inv(q, 1) = gi12;
        sk.g_inv(q, 2) = gi22;

        // Dual basis a^α = g^{αβ} a_β.
        sk.aup1.row(q) = gi11 * a1_q + gi12 * a2_q;
        sk.aup2.row(q) = gi12 * a1_q + gi22 * a2_q;

        if (order < 2) continue;

        const auto a11_q = sk.a11.row(q);
        const auto a12_q = sk.a12.row(q);
        const auto a22_q = sk.a22.row(q);

        // Second fundamental form b_αβ = a_{αβ} · a3.
        sk.b(q, 0) = a11_q.dot(a3_q.transpose());
        sk.b(q, 1) = a12_q.dot(a3_q.transpose());
        sk.b(q, 2) = a22_q.dot(a3_q.transpose());

        // Christoffels Γ^δ_{αβ} = a^δ · a_{αβ}, packed as
        // (Γ¹₁₁, Γ¹₁₂, Γ¹₂₂, Γ²₁₁, Γ²₁₂, Γ²₂₂).
        const auto aup1_q = sk.aup1.row(q);
        const auto aup2_q = sk.aup2.row(q);
        sk.christoffel(q, 0) = aup1_q.dot(a11_q);
        sk.christoffel(q, 1) = aup1_q.dot(a12_q);
        sk.christoffel(q, 2) = aup1_q.dot(a22_q);
        sk.christoffel(q, 3) = aup2_q.dot(a11_q);
        sk.christoffel(q, 4) = aup2_q.dot(a12_q);
        sk.christoffel(q, 5) = aup2_q.dot(a22_q);

        // Director derivatives a_{3,β} = -b^γ_β a_γ (Weingarten), with
        // b^γ_β = g^{γδ} b_{δβ}. Used by the membrane–bending coupling
        // term u_{,α}·a_{3,β} in the full Naghdi bending strain.
        const T b11 = sk.b(q, 0);
        const T b12 = sk.b(q, 1);
        const T b22 = sk.b(q, 2);
        const T bup11 = gi11 * b11 + gi12 * b12;   // b^1_1
        const T bup12 = gi11 * b12 + gi12 * b22;   // b^1_2
        const T bup21 = gi12 * b11 + gi22 * b12;   // b^2_1
        const T bup22 = gi12 * b12 + gi22 * b22;   // b^2_2
        sk.a3_1.row(q) = -(bup11 * a1_q + bup21 * a2_q);
        sk.a3_2.row(q) = -(bup12 * a1_q + bup22 * a2_q);
    }
    return sk;
}

template <std::floating_point T, std::size_t d>
std::tuple<ColMatrix<T, 3>, ColMatrix<T, 3>, ColMatrix<T, 3>,
           ColMatrix<T, 3>, ColMatrix<T, 3>, Vector<T>>
Patch<T, d>::eval_surface_geometry(const ColMatrix<T, d>& points,
                                   Index span) const requires(d == 2)
{
    SurfaceKinematics<T> sk = this->eval_surface_kinematics(points, span, 2);
    return {std::move(sk.a1), std::move(sk.a2), std::move(sk.a3),
            std::move(sk.g_inv), std::move(sk.b), std::move(sk.jac)};
}

template <std::floating_point T, std::size_t d>
std::vector<Matrix<T>> Patch<T, d>::eval_covariant_derivatives(
    const ColMatrix<T, d>& points,
    Index span,
    std::size_t order_in) const requires(d == 2)
{
    const Index order = std::max(Index(1), std::min(static_cast<Index>(order_in), Index(3)));

    SurfaceKinematics<T> sk = this->eval_surface_kinematics(points, span, order);
    const Index Q = points.rows();
    const Index S = order + 1;

    const Matrix<T>& N   = sk.basis_derivs[0 * S + 0];
    const Matrix<T>& N_u = sk.basis_derivs[1 * S + 0];
    const Matrix<T>& N_v = sk.basis_derivs[0 * S + 1];
    const Index K = N.cols();

    const std::size_t out_size = (order == 1) ? 3 : (order == 2 ? 6 : 12);
    std::vector<Matrix<T>> result(out_size);
    result[0] = N;
    result[1] = N_u;
    result[2] = N_v;

    if (order < 2) return result;

    const Matrix<T>& N_uu = sk.basis_derivs[2 * S + 0];
    const Matrix<T>& N_uv = sk.basis_derivs[1 * S + 1];
    const Matrix<T>& N_vv = sk.basis_derivs[0 * S + 2];

    result[3] = Matrix<T>(Q, K);
    result[4] = Matrix<T>(Q, K);
    result[5] = Matrix<T>(Q, K);

    if (order >= 3) {
        for (Index k = 6; k < 12; ++k) result[k] = Matrix<T>(Q, K);
    }

    for (Index q = 0; q < Q; ++q) {
        // Christoffels Γ^δ_{αβ} from the kinematics bundle. Packing:
        // (Γ¹₁₁, Γ¹₁₂, Γ¹₂₂, Γ²₁₁, Γ²₁₂, Γ²₂₂).
        const T G1_11 = sk.christoffel(q, 0);
        const T G1_12 = sk.christoffel(q, 1);
        const T G1_22 = sk.christoffel(q, 2);
        const T G2_11 = sk.christoffel(q, 3);
        const T G2_12 = sk.christoffel(q, 4);
        const T G2_22 = sk.christoffel(q, 5);

        // Covariant 2nd: N_{|αβ} = N_{,αβ} - Γ^γ_{αβ} N_{,γ}.
        result[3].row(q) = N_uu.row(q) - G1_11 * N_u.row(q) - G2_11 * N_v.row(q);
        result[4].row(q) = N_uv.row(q) - G1_12 * N_u.row(q) - G2_12 * N_v.row(q);
        result[5].row(q) = N_vv.row(q) - G1_22 * N_u.row(q) - G2_22 * N_v.row(q);

        if (order < 3) continue;

        // ===== Christoffel derivatives Γ^δ_{αβ,γ} =====
        const auto a1_q   = sk.a1.row(q);
        const auto a2_q   = sk.a2.row(q);
        const auto a11_q  = sk.a11.row(q);
        const auto a12_q  = sk.a12.row(q);
        const auto a22_q  = sk.a22.row(q);
        const auto a111_q = sk.a111.row(q);
        const auto a112_q = sk.a112.row(q);
        const auto a122_q = sk.a122.row(q);
        const auto a222_q = sk.a222.row(q);
        const auto aup1   = sk.aup1.row(q);
        const auto aup2   = sk.aup2.row(q);

        const T gi11 = sk.g_inv(q, 0);
        const T gi12 = sk.g_inv(q, 1);
        const T gi22 = sk.g_inv(q, 2);

        // Metric derivatives g_{αβ,γ} = a_α · a_{βγ} + a_{αγ} · a_β.
        const T g11_1 = T(2) * a1_q.dot(a11_q);
        const T g11_2 = T(2) * a1_q.dot(a12_q);
        const T g12_1 = a1_q.dot(a12_q) + a11_q.dot(a2_q);
        const T g12_2 = a1_q.dot(a22_q) + a12_q.dot(a2_q);
        const T g22_1 = T(2) * a2_q.dot(a12_q);
        const T g22_2 = T(2) * a2_q.dot(a22_q);

        // Inverse-metric derivatives g^{δσ}_{,γ} = -g^{δi} g^{σj} g_{ij,γ}.
        auto dg_inv = [&](T gd1, T gd2, T gs1, T gs2,
                          T g11g, T g12g, T g22g) -> T {
            return -(gd1 * gs1 * g11g
                   + (gd1 * gs2 + gd2 * gs1) * g12g
                   + gd2 * gs2 * g22g);
        };
        const T gi11_1 = dg_inv(gi11, gi12, gi11, gi12, g11_1, g12_1, g22_1);
        const T gi11_2 = dg_inv(gi11, gi12, gi11, gi12, g11_2, g12_2, g22_2);
        const T gi12_1 = dg_inv(gi11, gi12, gi12, gi22, g11_1, g12_1, g22_1);
        const T gi12_2 = dg_inv(gi11, gi12, gi12, gi22, g11_2, g12_2, g22_2);
        const T gi22_1 = dg_inv(gi12, gi22, gi12, gi22, g11_1, g12_1, g22_1);
        const T gi22_2 = dg_inv(gi12, gi22, gi12, gi22, g11_2, g12_2, g22_2);

        // Dual-basis derivatives a^δ_{,γ} = g^{δσ}_{,γ} a_σ + g^{δσ} a_{σγ}.
        Eigen::RowVector<T, 3> aup1_1 = gi11_1 * a1_q + gi12_1 * a2_q + gi11 * a11_q + gi12 * a12_q;
        Eigen::RowVector<T, 3> aup1_2 = gi11_2 * a1_q + gi12_2 * a2_q + gi11 * a12_q + gi12 * a22_q;
        Eigen::RowVector<T, 3> aup2_1 = gi12_1 * a1_q + gi22_1 * a2_q + gi12 * a11_q + gi22 * a12_q;
        Eigen::RowVector<T, 3> aup2_2 = gi12_2 * a1_q + gi22_2 * a2_q + gi12 * a12_q + gi22 * a22_q;

        // Γ^δ_{αβ,γ} = a^δ_{,γ} · a_{αβ} + a^δ · a_{αβγ}.
        const T G1_11_1 = aup1_1.dot(a11_q) + aup1.dot(a111_q);
        const T G1_11_2 = aup1_2.dot(a11_q) + aup1.dot(a112_q);
        const T G1_12_1 = aup1_1.dot(a12_q) + aup1.dot(a112_q);
        const T G1_12_2 = aup1_2.dot(a12_q) + aup1.dot(a122_q);
        const T G1_22_1 = aup1_1.dot(a22_q) + aup1.dot(a122_q);
        const T G1_22_2 = aup1_2.dot(a22_q) + aup1.dot(a222_q);
        const T G2_11_1 = aup2_1.dot(a11_q) + aup2.dot(a111_q);
        const T G2_11_2 = aup2_2.dot(a11_q) + aup2.dot(a112_q);
        const T G2_12_1 = aup2_1.dot(a12_q) + aup2.dot(a112_q);
        const T G2_12_2 = aup2_2.dot(a12_q) + aup2.dot(a122_q);
        const T G2_22_1 = aup2_1.dot(a22_q) + aup2.dot(a122_q);
        const T G2_22_2 = aup2_2.dot(a22_q) + aup2.dot(a222_q);

        // ===== Covariant 3rd derivatives =====
        // f_{|αβγ} = ∂_γ f_{|αβ} - Γ^σ_{γα} f_{|σβ} - Γ^σ_{γβ} f_{|ασ}
        // ∂_γ f_{|αβ} = f_{,αβγ} - Γ^δ_{αβ,γ} f_{,δ} - Γ^δ_{αβ} f_{,δγ}
        const auto N_uuu = sk.basis_derivs[3 * S + 0].row(q);
        const auto N_uuv = sk.basis_derivs[2 * S + 1].row(q);
        const auto N_uvv = sk.basis_derivs[1 * S + 2].row(q);
        const auto N_vvv = sk.basis_derivs[0 * S + 3].row(q);

        const auto Nu  = N_u.row(q);
        const auto Nv  = N_v.row(q);
        const auto Nuu = N_uu.row(q);
        const auto Nuv = N_uv.row(q);
        const auto Nvv = N_vv.row(q);
        const auto f11 = result[3].row(q);
        const auto f12 = result[4].row(q);
        const auto f22 = result[5].row(q);

        // (αβ,γ) = (11,1): correction terms both = -Γ^σ_{11} f_{|1σ}, doubled.
        result[6].row(q) = N_uuu
            - G1_11_1 * Nu - G2_11_1 * Nv
            - G1_11   * Nuu - G2_11   * Nuv
            - T(2) * (G1_11 * f11 + G2_11 * f12);

        // (αβ,γ) = (11,2): both correction terms = -Γ^σ_{12} f_{|1σ}, doubled.
        result[7].row(q) = N_uuv
            - G1_11_2 * Nu - G2_11_2 * Nv
            - G1_11   * Nuv - G2_11   * Nvv
            - T(2) * (G1_12 * f11 + G2_12 * f12);

        // (αβ,γ) = (12,1): -Γ^σ_{1,β=2} f_{|1σ} - Γ^σ_{1,α=1} f_{|2σ}.
        result[8].row(q) = N_uuv
            - G1_12_1 * Nu - G2_12_1 * Nv
            - G1_12   * Nuu - G2_12   * Nuv
            - (G1_12 * f11 + G2_12 * f12)
            - (G1_11 * f12 + G2_11 * f22);

        // (αβ,γ) = (12,2): -Γ^σ_{2,β=2} f_{|1σ} - Γ^σ_{2,α=1} f_{|2σ}.
        result[9].row(q) = N_uvv
            - G1_12_2 * Nu - G2_12_2 * Nv
            - G1_12   * Nuv - G2_12   * Nvv
            - (G1_22 * f11 + G2_22 * f12)
            - (G1_12 * f12 + G2_12 * f22);

        // (αβ,γ) = (22,1): both correction terms = -Γ^σ_{12} f_{|2σ}, doubled.
        result[10].row(q) = N_uvv
            - G1_22_1 * Nu - G2_22_1 * Nv
            - G1_22   * Nuu - G2_22   * Nuv
            - T(2) * (G1_12 * f12 + G2_12 * f22);

        // (αβ,γ) = (22,2): both correction terms = -Γ^σ_{22} f_{|2σ}, doubled.
        result[11].row(q) = N_vvv
            - G1_22_2 * Nu - G2_22_2 * Nv
            - G1_22   * Nuv - G2_22   * Nvv
            - T(2) * (G1_22 * f12 + G2_22 * f22);
    }
    return result;
}

template class Patch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 2>;
#endif

} // namespace pyck
