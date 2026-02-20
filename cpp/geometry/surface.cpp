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

    auto Nu = basis_[0]->eval(u, order);
    auto Nv = basis_[1]->eval(v, order);

    const std::size_t n_u = basis_[0]->num_basis();
    const std::size_t n_v = basis_[1]->num_basis();

    std::vector<std::vector<Eigen::MatrixXd>> result(order + 1);
    for (std::size_t i = 0; i <= order; ++i) {
        result[i].resize(order + 1 - i);
    }

    for (std::size_t i = 0; i <= order; ++i) {
        for (std::size_t j = 0; j <= order - i; ++j) {
            Eigen::MatrixXd R(Q, n_u * n_v);

            for (std::size_t p = 0; p < M; ++p) {
                for (std::size_t q = 0; q < N; ++q) {
                    std::size_t row = p * N + q;
                    for (std::size_t a = 0; a < n_u; ++a) {
                        double nu_val = Nu[i](p, a);
                        for (std::size_t b = 0; b < n_v; ++b) {
                            R(row, a * n_v + b) = nu_val * Nv[j](q, b);
                        }
                    }
                }
            }

            result[i][j] = std::move(R);
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
