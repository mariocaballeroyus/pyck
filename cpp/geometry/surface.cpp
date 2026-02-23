#include "surface.hpp"

namespace pyck
{

std::vector<std::vector<Eigen::MatrixXd>> SurfacePatch::tensor_basis(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    std::size_t order) const
{
    const std::size_t M = u.size();
    const std::size_t N = v.size();
    const std::size_t Q = M * N;

    // Build the M*N × 2 parameter grid (v runs fastest)
    Eigen::MatrixXd params(Q, 2);
    for (std::size_t p = 0; p < M; ++p) {
        for (std::size_t q = 0; q < N; ++q) {
            params(p * N + q, 0) = u(p);
            params(p * N + q, 1) = v(q);
        }
    }

    // Allocate the triangular result[i][j] with i + j <= order
    std::vector<std::vector<Eigen::MatrixXd>> result(order + 1);
    for (std::size_t i = 0; i <= order; ++i) {
        result[i].resize(order + 1 - i);
    }

    if (order == 0) {
        // Fast path: only basis values, no derivatives
        result[0][0] = tensor_product_.eval(params);
    } else {
        // eval_derivs returns (order+1)*(order+1) matrices in flat order:
        //   index = i_u * (order+1) + i_v
        // where i_u is the derivative order in u and i_v in v.
        auto derivs = tensor_product_.eval_derivs(params, {order, order});

        for (std::size_t i = 0; i <= order; ++i) {
            for (std::size_t j = 0; j <= order - i; ++j) {
                result[i][j] = std::move(derivs[i * (order + 1) + j]);
            }
        }
    }

    return result;
}


std::vector<std::vector<Eigen::MatrixXd>> SurfacePatch::eval(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    std::size_t order) const
{
    auto R = tensor_basis(u, v, order);

    // Multiply each basis matrix by the control points
    for (std::size_t i = 0; i <= order; ++i) {
        for (std::size_t j = 0; j <= order - i; ++j) {
            R[i][j] = R[i][j] * control_points_;
        }
    }

    return R;
}

std::pair<std::vector<Eigen::MatrixXd>, Eigen::VectorXd>
SurfacePatch::jacobian(const Eigen::VectorXd& u,
                       const Eigen::VectorXd& v) const
{
    // Evaluate first-order parametric derivatives
    auto derivs = eval(u, v, 1);
    const auto& dSdu = derivs[1][0];  // (Q × gdim)
    const auto& dSdv = derivs[0][1];  // (Q × gdim)

    const Eigen::Index Q = dSdu.rows();
    const Eigen::Index d = static_cast<Eigen::Index>(gdim_);

    std::vector<Eigen::MatrixXd> jacobians(Q);
    Eigen::VectorXd dets(Q);

    for (Eigen::Index q = 0; q < Q; ++q) {
        // J is (gdim × 2)
        Eigen::MatrixXd J(d, 2);
        J.col(0) = dSdu.row(q).transpose();
        J.col(1) = dSdv.row(q).transpose();
        jacobians[q] = J;

        // det = sqrt(det(J^T J))
        Eigen::Matrix2d JtJ = J.transpose() * J;
        dets(q) = std::sqrt(JtJ.determinant());
    }

    return {std::move(jacobians), std::move(dets)};
}

} // namespace pyck
