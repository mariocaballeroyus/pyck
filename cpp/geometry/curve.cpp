#include "curve.hpp"
#include "bspline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
CurvePatch<T>::CurvePatch(Ptr<const Basis<T>> basis_u, const ColMatrix<T, 3>& control_pts)
    : Patch<T, 1>(control_pts),
      tensor_product_(std::move(basis_u)),
      dof_mapper_({tensor_product_.basis(0).num_basis()},
                  {tensor_product_.basis(0).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("CurvePatch: "
                                    "Control points must be embedded in 3D space."
        );
    }

    const Index expected_n = tensor_product_.basis(0).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n) {
        throw std::invalid_argument("CurvePatch: "
                                    "Dimension mismatch. Expected "
                                    + std::to_string(expected_n) + " control points"
        );
    }

}

template <std::floating_point T>
std::vector<Matrix<T>> CurvePatch<T>::eval_basis_functions(const ColMatrix<T, 1>& points,
                                                           const std::array<Index, 1>& spans,
                                                           Index order) const
{
    return tensor_product_.eval_derivs(points, spans, order);
}

template <std::floating_point T>
std::pair<std::vector<Matrix<T>>, Vector<T>> CurvePatch<T>::eval_shape_functions(
    const ColMatrix<T, 1>& points,
    const std::array<Index, 1>& spans,
    Index order) const
{
    order = std::max(Index(1), std::min(order, Index(3)));
    const Index Q = points.rows();

    auto basis_derivs = tensor_product_.eval_derivs(points, spans, order);
    const Index K = basis_derivs[0].cols();

    auto act_pts = this->active_control_pts(spans);
    ColMatrix<T, 3> x_u = basis_derivs[1] * act_pts; 
    ColMatrix<T, 3> x_uu = (order >= 2) ? (basis_derivs[2] * act_pts) : ColMatrix<T, 3>();
    ColMatrix<T, 3> x_uuu = (order >= 3) ? (basis_derivs[3] * act_pts) : ColMatrix<T, 3>();

    std::vector<Matrix<T>> result(order + 1);
    Vector<T> jacobian(Q);

    result[0] = basis_derivs[0];
    if (order >= 1) result[1] = Matrix<T>(Q, K);
    if (order >= 2) result[2] = Matrix<T>(Q, K);
    if (order >= 3) result[3] = Matrix<T>(Q, K);

    for (Index q = 0; q < Q; ++q) 
    {
        // 1st Order (Tangent component)
        const T g11 = x_u.row(q).squaredNorm(); 
        const T J = std::sqrt(g11);

        jacobian(q) = J;
        result[1].row(q) = basis_derivs[1].row(q) / J;

        if (order < 2) continue;

        // 2nd Order (Curvature component)
        const T Gamma = x_u.row(q).dot(x_uu.row(q)) / g11;

        result[2].row(q) = (basis_derivs[2].row(q) - Gamma * basis_derivs[1].row(q)) / g11;

        if (order < 3) continue;

        // 3rd Order (Jerk component)
        const T Gamma_u = (x_uu.row(q).squaredNorm() + x_u.row(q).dot(x_uuu.row(q))) / g11 - static_cast<T>(2.0) * (Gamma * Gamma);
        const T J3 = g11 * J; 
        
        result[3].row(q) = (basis_derivs[3].row(q) - 
                         static_cast<T>(3.0) * Gamma * basis_derivs[2].row(q) + 
                         (static_cast<T>(2.0) * (Gamma * Gamma) - Gamma_u) * basis_derivs[1].row(q)) / J3;
    }

    return {std::move(result), std::move(jacobian)};
}

template <std::floating_point T>
std::vector<ColMatrix<T, 3>> CurvePatch<T>::eval_geometry(const ColMatrix<T, 1>& points,
                                                          const std::array<Index, 1>& spans,
                                                          Index order) const
{
    auto basis_derivs = tensor_product_.eval_derivs(points, spans, order);
    auto act_pts = this->active_control_pts(spans);

    std::vector<ColMatrix<T, 3>> result(order + 1);
    for (Index i = 0; i <= order; ++i) {
        result[i] = basis_derivs[i] * act_pts;
    }
    return result;
}

// === Factory Methods ================================================================

template <std::floating_point T>
static std::vector<T> greville_abscissae(Ptr<const Basis<T>> bs)
{
    Index n = bs->num_basis();
    Index p = bs->degree();
    
    auto bspline = std::dynamic_pointer_cast<const BSpline<T>>(bs);
    if (!bspline) {
        throw std::runtime_error(
            "Greville abscissae requires a BSpline basis implementation."
        );
    }
    
    const auto& knots_vec = bspline->knots();
    std::vector<T> xi(n);
    for (Index i = 0; i < n; ++i) {
        T sum = 0.0;
        for (Index j = 1; j <= p; ++j)
            sum += knots_vec[i + j];
        xi[i] = sum / static_cast<T>(p);
    }
    return xi;
}

template <std::floating_point T> 
CurvePatch<T> line_segment(Ptr<const pyck::Basis<T>> basis, 
                           T length)
{
    std::vector<T> xi = greville_abscissae(basis);
    Index n = basis->num_basis();

    ColMatrix<T, 3> P = ColMatrix<T, 3>::Zero(n, 3);

    T u_min = xi.front();
    T u_max = xi.back();
    T u_range = u_max - u_min;

    if (u_range < 1e-14) {
        u_range = 1.0; 
    }

    for (Index i = 0; i < n; ++i) {
        T normalized_u = (xi[i] - u_min) / u_range;
        P(i, 0) = length * normalized_u;
    }

    return CurvePatch<T>(basis, P);
}

// === Template Instantiations ========================================================

template class CurvePatch<double>;
template CurvePatch<double> line_segment<double>(Ptr<const Basis<double>>, double);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class CurvePatch<float>;
template CurvePatch<float> line_segment<float>(Ptr<const Basis<float>>, float);
#endif

} // namespace pyck
