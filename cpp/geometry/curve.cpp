#include "curve.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace pyck
{

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

CurvePatch CurvePatch::line_segment(BSpline* basis, double length, size_t gdim)
{
    auto shared_basis = std::shared_ptr<BSpline>(basis);

    std::vector<double> xi = greville_abscissae(*shared_basis);
    std::size_t n = shared_basis->num_basis();

    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, gdim);

    double u_min = xi.front();
    double u_max = xi.back();
    double u_range = u_max - u_min;

    if (u_range < 1e-14) {
        u_range = 1.0; 
    }

    for (std::size_t i = 0; i < n; ++i) {
        double normalized_u = (xi[i] - u_min) / u_range;
        P(i, 0) = length * normalized_u;
    }

    return CurvePatch(gdim, std::move(shared_basis), P);
}

std::vector<Eigen::MatrixXd> CurvePatch::eval_basis_functions(const Eigen::MatrixXd& points, 
                                                               std::size_t order) const
{
    return tensor_product_.eval_derivs(points, {order});
}

std::vector<Eigen::MatrixXd> CurvePatch::eval_shape_functions(const Eigen::MatrixXd& params, 
                                                               std::size_t order) const 
{
    order = std::max<std::size_t>(1, std::min<std::size_t>(order, 3));
    const Eigen::Index Q = params.rows();
    const std::size_t K  = tensor_product_.num_basis();

    auto basis_derivs = tensor_product_.eval_derivs(params, {order});
    Eigen::MatrixXd x_u = basis_derivs[1] * control_pts_; 
    Eigen::MatrixXd x_uu = (order >= 2) ? (basis_derivs[2] * control_pts_) : Eigen::MatrixXd();
    Eigen::MatrixXd x_uuu = (order >= 3) ? (basis_derivs[3] * control_pts_) : Eigen::MatrixXd();

    std::vector<Eigen::MatrixXd> result(order + 1);

    result[0] = basis_derivs[0];
    if (order >= 1) result[1] = Eigen::MatrixXd(Q, K);
    if (order >= 2) result[2] = Eigen::MatrixXd(Q, K);
    if (order >= 3) result[3] = Eigen::MatrixXd(Q, K);

    for (Eigen::Index q = 0; q < Q; ++q) {
        // 1st Order (Tangent component)
        const double g11 = x_u.row(q).squaredNorm(); 
        const double J = std::sqrt(g11);
        
        result[1].row(q) = basis_derivs[1].row(q) / J;

        if (order < 2) continue;

        // 2nd Order (Curvature component)
        const double Gamma = x_u.row(q).dot(x_uu.row(q)) / g11;
        
        result[2].row(q) = (basis_derivs[2].row(q) - Gamma * basis_derivs[1].row(q)) / g11;

        if (order < 3) continue;

        // 3rd Order (Jerk component)
        const double Gamma_u = (x_uu.row(q).squaredNorm() + x_u.row(q).dot(x_uuu.row(q))) / g11 - 2.0 * (Gamma * Gamma);
        const double J3 = g11 * J; 
        
        result[3].row(q) = (basis_derivs[3].row(q) - 
                         3.0 * Gamma * basis_derivs[2].row(q) + 
                         (2.0 * (Gamma * Gamma) - Gamma_u) * basis_derivs[1].row(q)) / J3;
    }

    return result;
}

std::vector<Eigen::MatrixXd> CurvePatch::eval_geometry(const Eigen::MatrixXd& points,
                                                        std::size_t order) const
{
    auto basis_derivs = tensor_product_.eval_derivs(points, {order});
    std::vector<Eigen::MatrixXd> result(order + 1);
    for (std::size_t i = 0; i <= order; ++i) {
        result[i] = basis_derivs[i] * control_pts_;
    }
    return result;
}

Eigen::VectorXd CurvePatch::eval_jacobian(const Eigen::MatrixXd& points) const
{
    auto basis_derivs = tensor_product_.eval_derivs(points, {std::size_t(1)});
    Eigen::MatrixXd x_u = basis_derivs[1] * control_pts_;
    Eigen::VectorXd jac(points.rows());
    for (Eigen::Index q = 0; q < points.rows(); ++q) {
        jac(q) = x_u.row(q).norm();
    }
    return jac;
}

} // namespace pyck
