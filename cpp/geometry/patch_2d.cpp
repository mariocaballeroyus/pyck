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

template class Patch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 2>;
#endif

} // namespace pyck
