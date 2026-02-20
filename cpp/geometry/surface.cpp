#include "surface.hpp"

namespace pyck
{

std::vector<std::vector<Eigen::MatrixXd>> SurfacePatch::eval(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    std::size_t order) const
{
    const std::size_t M = u.size();  // number of u points
    const std::size_t N = v.size();  // number of v points
    const std::size_t Q = M * N;     // total evaluation points

    // --- Evaluate 1-D basis functions and derivatives ---
    // Nu[i] is the i-th u-derivative: (M x n_u)
    // Nv[j] is the j-th v-derivative: (N x n_v)
    auto Nu = basis_[0]->eval(u, order);   // size order+1
    auto Nv = basis_[1]->eval(v, order);   // size order+1

    const std::size_t n_u = basis_[0]->num_basis();
    const std::size_t n_v = basis_[1]->num_basis();

    // --- Allocate output ---
    // result[i][j] exists for 0 <= i+j <= order
    std::vector<std::vector<Eigen::MatrixXd>> result(order + 1);
    for (std::size_t i = 0; i <= order; ++i) {
        result[i].resize(order + 1 - i);
    }

    // --- Build tensor-product basis and map to physical space ---
    // For each pair (i, j) with i+j <= order, assemble the (Q x n_u*n_v)
    // tensor-product basis matrix from Nu[i] and Nv[j], then multiply by
    // the control-point matrix.
    //
    // Control points are stored as (n_u * n_v) x gdim with v running fastest:
    //   row index = a * n_v + b   (a = u-index, b = v-index)
    //
    // The tensor-product basis at grid point (p, q) — row p*N + q:
    //   R(p*N+q, a*n_v+b) = Nu[i](p,a) * Nv[j](q,b)
    //
    // Physical coordinates:  X = R * P

    for (std::size_t i = 0; i <= order; ++i) {
        for (std::size_t j = 0; j <= order - i; ++j) {
            // Build the tensor-product matrix R: (Q x n_u*n_v)
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

            result[i][j] = R * control_points_;
        }
    }

    return result;
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
