#include "patch.hpp"
#include "bspline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "../quadrature/quadrature.hpp"

namespace pyck
{

// === Constrained Constructors and Methods for d=1 ===========================

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u, const ColMatrix<T, 3>& control_pts) requires(d == 1)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u)),
      dof_mapper_({tensor_product_.basis(0).num_basis()}, {tensor_product_.basis(0).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Control points must be embedded in 3D space.");
    }
    const Index expected_n = this->tensor_product_.basis(0).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());
    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Dimension mismatch.");
    }

    // Initialize Greville points and spans
    Vector<T> gu = this->tensor_product_.basis(0).greville_abscissae();
    greville_points_.resize(gu.size(), 1);
    greville_spans_.resize(gu.size());
    for (Index i = 0; i < static_cast<Index>(gu.size()); ++i) {
        greville_points_(i, 0) = gu[i];
        greville_spans_[i] = this->tensor_product_.basis(0).find_span(gu[i]);
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
Patch<T, d>::eval_shape_functions(const ColMatrix<T, d>& points, Index span, std::size_t order_in) const requires(d == 1)
{
    std::array<Index, 1> spans = {span};
    Index order = std::max(Index(1), std::min(static_cast<Index>(order_in), Index(3)));
    const Index Q = points.rows();

    auto basis_derivs = this->tensor_product().eval_on_span(points, spans, order);
    const Index K = basis_derivs[0].cols();

    auto act_pts = this->active_control_pts(spans);

    // Tangent vector a_1 = ∂x/∂ξ^1
    ColMatrix<T, 3> x_u = basis_derivs[1] * act_pts; 
    ColMatrix<T, 3> x_uu = (order >= 2) ? (basis_derivs[2] * act_pts) : ColMatrix<T, 3>();
    ColMatrix<T, 3> x_uuu = (order >= 3) ? (basis_derivs[3] * act_pts) : ColMatrix<T, 3>();

    std::vector<Matrix<T>> result(order + 1);
    // Compute Jacobian (length element) at each quadrature point
    Vector<T> jacobian(Q);

    result[0] = basis_derivs[0];
    if (order >= 1) result[1] = Matrix<T>(Q, K);
    if (order >= 2) result[2] = Matrix<T>(Q, K);
    if (order >= 3) result[3] = Matrix<T>(Q, K);

    for (Index q = 0; q < Q; ++q) 
    {
        // 1st Order (Tangent component)
        // a_1 = x_{,u}
        // J = ||a_1||
        const T g11 = x_u.row(q).squaredNorm(); 
        const T J = std::sqrt(g11);
        jacobian(q) = J;
        result[1].row(q) = basis_derivs[1].row(q) / J;

        if (order < 2) continue;

        // 2nd Order (Curvature component)
        // Γ = (a_1 · a_{11}) / g11
        // N_{;uu} = (N_{,uu} - Γ * N_{,u}) / g11
        const T Gamma = x_u.row(q).dot(x_uu.row(q)) / g11;
        result[2].row(q) = (basis_derivs[2].row(q) - Gamma * basis_derivs[1].row(q)) / g11;

        if (order < 3) continue;

        // 3rd Order (Jerk component)
        // Γ_u = (||x_{,uu}||^2 + x_{,u} · x_{,uuu}) / g11 - 2 * Γ^2
        // N_{;uuu} = (N_{,uuu} - 3*Γ*N_{;uu} - Γ_u*N_{,u}) / J
        const T Gamma_u = (x_uu.row(q).squaredNorm() + x_u.row(q).dot(x_uuu.row(q))) / g11 - static_cast<T>(2.0) * (Gamma * Gamma);
        const T J3 = g11 * J; 
        
        result[3].row(q) = (basis_derivs[3].row(q) - 
                        static_cast<T>(3.0) * Gamma * basis_derivs[2].row(q) + 
                        (static_cast<T>(2.0) * (Gamma * Gamma) - Gamma_u) * basis_derivs[1].row(q)) / J3;
    }
    return {std::move(result), std::move(jacobian)};
}


template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::eval_geometry(const ColMatrix<T, d>& points, Index span) const requires(d == 1)
{
    // Evaluate via the basis (works uniformly for B-spline and NURBS,
    // as NURBS::eval_on_span already returns rational shape functions).
    std::array<Index, 1> span_arr = {span};
    auto basis_fns = this->tensor_product().eval_on_span(points, span_arr, 0);
    auto act_pts = this->active_control_pts(span_arr);
    return basis_fns[0] * act_pts;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::eval_physical_points(const QuadratureRule<T, d>& quadrature) const requires(d == 1)
{
    const Index Q = static_cast<Index>(quadrature.num_points());
    const Index num_spans = static_cast<Index>(this->basis(0).knot_vector().num_spans());

    ColMatrix<T, 3> result(num_spans * Q, 3);
    Index out = 0;

    for (Index s = 0; s < num_spans; ++s)
    {
        auto [lo, hi] = this->basis(0).knot_vector().span_bounds(s);
        if (std::abs(hi - lo) < static_cast<T>(1e-14))
            continue; // skip zero-volume clamped span

        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        auto geom = this->eval_geometry(mapped_pts, s);
        result.block(out, 0, Q, 3) = geom;
        out += Q;
    }
    result.conservativeResize(out, Eigen::NoChange);
    return result;
}

template <std::floating_point T, std::size_t d>
std::array<ColMatrix<T, 3>, d> Patch<T, d>::eval_tangent(
    const std::vector<Matrix<T>>& N,
    const ColMatrix<T, 3>& act_pts) const requires(d == 1)
{
    std::array<ColMatrix<T, 3>, 1> tangents;
    if (N.size() < 2) throw std::runtime_error("eval_tangent: N must contain at least 2 entries (values and 1st derivative)");
    tangents[0] = N[1] * act_pts;
    return tangents;
}

template class Patch<double, 1>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 1>;
#endif

} // namespace pyck
