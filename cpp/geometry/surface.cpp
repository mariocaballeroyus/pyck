#include "surface.hpp"

namespace pyck
{

/**
 * @brief Compute Greville abscissae for a B-spline basis.
 *
 * The i-th Greville point is the average of the p knots following ξ_i:
 *   ξ̄_i = (1/p) Σ_{j=1}^{p} ξ_{i+j}
 */
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
    // Bernstein control values for f(t) = t:  {0, 0.5, 1}
    // Bernstein control values for f(t) = t²: {0, 0,   1}
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

Eigen::MatrixXd SurfacePatch::eval(const Eigen::MatrixXd& params) const
{
    return tensor_product_.eval(params) * control_points_;
}

std::vector<std::vector<Eigen::MatrixXd>> SurfacePatch::eval_derivs(
    const Eigen::MatrixXd& params,
    std::size_t order) const
{
    // Allocate the triangular result[i][j] with i + j <= order
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

std::vector<Eigen::MatrixXd> SurfacePatch::jacobian(const Eigen::MatrixXd& params) const
{
    auto derivs = eval_derivs(params, 1);
    const auto& dSdu = derivs[1][0];  // (Q × gdim)
    const auto& dSdv = derivs[0][1];  // (Q × gdim)

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
