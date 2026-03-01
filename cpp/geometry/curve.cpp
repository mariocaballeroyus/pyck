#include "curve.hpp"
#include "bspline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
static std::vector<T> greville_abscissae(std::shared_ptr<const Basis<T>> bs)
{
    std::size_t n = bs->num_basis();
    std::size_t p = bs->degree();
    
    auto bspline = std::dynamic_pointer_cast<const BSpline<T>>(bs);
    if (!bspline) {
        throw std::runtime_error("Greville abscissae requires a BSpline basis implementation.");
    }
    
    const auto& knots_vec = bspline->knots();
    std::vector<T> xi(n);
    for (std::size_t i = 0; i < n; ++i) {
        T sum = 0.0;
        for (std::size_t j = 1; j <= p; ++j)
            sum += knots_vec[i + j];
        xi[i] = sum / static_cast<T>(p);
    }
    return xi;
}

template <std::floating_point T> 
CurvePatch<T> CurvePatch<T>::line_segment(std::shared_ptr<const pyck::Basis<T>> basis, 
                                  ScalarType length)
{
    std::vector<T> xi = greville_abscissae(basis);
    std::size_t n = basis->num_basis();

    ColMatrix<T, 3> P = ColMatrix<T, 3>::Zero(n, 3);

    ScalarType u_min = xi.front();
    ScalarType u_max = xi.back();
    ScalarType u_range = u_max - u_min;

    if (u_range < 1e-14) {
        u_range = 1.0; 
    }

    for (std::size_t i = 0; i < n; ++i) {
        ScalarType normalized_u = (xi[i] - u_min) / u_range;
        P(i, 0) = length * normalized_u;
    }

    return CurvePatch<T>(basis, P);
}

template <std::floating_point T>
std::vector<Matrix<T>> CurvePatch<T>::eval_basis_functions(const ColMatrix<T, 1>& points, 
                                                       std::size_t order) const
{
    return tensor_product_.eval_derivs(points, {order});
}

template <std::floating_point T>
std::vector<Matrix<T>> CurvePatch<T>::eval_shape_functions(const ColMatrix<T, 1>& params, 
                                                       std::size_t order) const
{
    order = std::max<std::size_t>(1, std::min<std::size_t>(order, 3));
    const Eigen::Index Q = params.rows();
    const std::size_t K  = tensor_product_.num_basis();

    auto basis_derivs = tensor_product_.eval_derivs(params, {order});
    ColMatrix<T, 3> x_u = basis_derivs[1] * this->control_pts_; 
    ColMatrix<T, 3> x_uu = (order >= 2) ? (basis_derivs[2] * this->control_pts_) : ColMatrix<T, 3>();
    ColMatrix<T, 3> x_uuu = (order >= 3) ? (basis_derivs[3] * this->control_pts_) : ColMatrix<T, 3>();

    std::vector<Matrix<T>> result(order + 1);

    result[0] = basis_derivs[0];
    if (order >= 1) result[1] = Matrix<T>(Q, K);
    if (order >= 2) result[2] = Matrix<T>(Q, K);
    if (order >= 3) result[3] = Matrix<T>(Q, K);

    for (Eigen::Index q = 0; q < Q; ++q) 
    {
        // 1st Order (Tangent component)
        const ScalarType g11 = x_u.row(q).squaredNorm(); 
        const ScalarType J = std::sqrt(g11);
        
        result[1].row(q) = basis_derivs[1].row(q) / J;

        if (order < 2) continue;

        // 2nd Order (Curvature component)
        const ScalarType Gamma = x_u.row(q).dot(x_uu.row(q)) / g11;
        
        result[2].row(q) = (basis_derivs[2].row(q) - Gamma * basis_derivs[1].row(q)) / g11;

        if (order < 3) continue;

        // 3rd Order (Jerk component)
        const ScalarType Gamma_u = (x_uu.row(q).squaredNorm() + x_u.row(q).dot(x_uuu.row(q))) / g11 - static_cast<ScalarType>(2.0) * (Gamma * Gamma);
        const ScalarType J3 = g11 * J; 
        
        result[3].row(q) = (basis_derivs[3].row(q) - 
                         static_cast<ScalarType>(3.0) * Gamma * basis_derivs[2].row(q) + 
                         (static_cast<ScalarType>(2.0) * (Gamma * Gamma) - Gamma_u) * basis_derivs[1].row(q)) / J3;
    }

    return result;
}

template <std::floating_point T>
std::vector<ColMatrix<T, 3>> CurvePatch<T>::eval_geometry(const ColMatrix<T, 1>& points,
                                                      std::size_t order) const
{
    auto basis_derivs = tensor_product_.eval_derivs(points, {order});
    std::vector<ColMatrix<T, 3>> result(order + 1);
    for (std::size_t i = 0; i <= order; ++i) {
        result[i] = basis_derivs[i] * this->control_pts_;
    }
    return result;
}

template <std::floating_point T>
Vector<T> CurvePatch<T>::eval_jacobian(const ColMatrix<T, 1>& points) const
{
    auto basis_derivs = tensor_product_.eval_derivs(points, {std::size_t(1)});
    ColMatrix<T, 3> x_u = basis_derivs[1] * this->control_pts_;
    Vector<T> jac(points.rows());
    for (Eigen::Index q = 0; q < points.rows(); ++q) {
        jac(q) = x_u.row(q).norm();
    }
    return jac;
}

// === Boundary Conditions ========================================================

template <std::floating_point T>
std::vector<Index> CurvePatch<T>::get_boundary_dofs(std::size_t param_dim, bool at_start) const
{
    if (param_dim != 0) {
        throw std::invalid_argument("param_dim must be 0 for a curve patch.");
    }

    std::vector<Index> clamped_dofs;
    Index n = this->basis(0).num_basis();
    
    if (at_start)
    {
        clamped_dofs.push_back(0);
        clamped_dofs.push_back(1);
    }
    else
    {
        clamped_dofs.push_back(n - 2);
        clamped_dofs.push_back(n - 1);
    }

    return clamped_dofs;
}

// === Template Instantiations ========================================================

template class CurvePatch<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class CurvePatch<float>;
#endif

} // namespace pyck
