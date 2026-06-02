#ifndef PYCK_PRIMITIVES_INTRINSIC_HPP
#define PYCK_PRIMITIVES_INTRINSIC_HPP

#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "../basis/tensor_product.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

namespace geometry::intrinsic
{

// === Position Derivatives ===========================================================

/**
 * @brief Fill per-order packed position-derivative output arrays from a basis and
 *        the active control points, up to a derivative order.
 * 
 * @param basis        Per-order basis-function values at the quadrature points. basis[k] 
 *                     holds the k-th order derivatives of every basis function.
 * @param active_pts   Active control points. 
 * @param position_out Output array for the evaluated derivatives, indexed by order.
 * @param max_order    Maximum derivative order to evaluate. Capped to the maximum 
 *                     degree of the provided basis.
 */
template <std::floating_point T, std::size_t d>
inline void compute_position_derivatives(const std::vector<Matrix<T>>& basis,
                                         const ColMatrix<T, 3>& active_pts,
                                         std::vector<ColMatrix<T, 3>>& position_out,
                                         Index max_order)
{
    const Index avail = static_cast<Index>(basis.size()) - 1;
    const Index order = (max_order < avail) ? max_order : avail;
    const Index num_qp = basis[0].cols();
    const Index num_b  = basis[0].rows();

    // Resize output array
    position_out.resize(order + 1);
    for (Index k = 0; k <= order; ++k) {
        const Index n_k = num_multi_indices<d>(k);
        position_out[k].resize(num_qp * n_k, 3);
    }

    for (Index q = 0; q < num_qp; ++q) {
        for (Index k = 0; k <= order; ++k) {
            const Index n_k = num_multi_indices<d>(k);
            auto slab = basis[k].col(q);
            for (Index packed = 0; packed < n_k; ++packed) {
                Eigen::Matrix<T, 1, 3> deriv_q = Eigen::Matrix<T, 1, 3>::Zero();
                for (Index b = 0; b < num_b; ++b) {
                    // R_{k} = Σ_b (∂^k N_b / ∂ξ_{k}) · P_b
                    deriv_q.noalias() += slab(b * n_k + packed) * active_pts.row(b);
                }
                position_out[k].row(packed * num_qp + q) = deriv_q;
            }
        }
    }
}

// === Metric =========================================================================

/**
 * @brief Compute covariant + contravariant metric and Jacobian from position
 *        derivatives at order 1 (covariant tangents a_α).
 * 
 * @param pos_derivs Per-order position derivatives sampled at Q quadrature points.
 * @param metric_out     Packed covariant metric components A_{αβ} at each point.
 * @param metric_inv_out Packed contravariant metric components A^{αβ} at each point.
 * @param jacobian_out   Jacobian at each point.
 */
template <std::floating_point T, std::size_t d>
inline void compute_metric(const std::vector<ColMatrix<T, 3>>& pos_derivs,
                           Matrix<T>& metric_out,
                           Matrix<T>& metric_inv_out,
                           Vector<T>& jacobian_out)
{
    // Aliases matching differential geometry notation
    const auto& R = pos_derivs;
    auto& A       = metric_out;
    auto& A_inv   = metric_inv_out;
    auto& J       = jacobian_out;
    
    constexpr Index n_metric = d * (d + 1) / 2;
    const Index num_qp = R[1].rows() / static_cast<Index>(d);

    // Resize output arrays
    A.resize(num_qp, n_metric);
    A_inv.resize(num_qp, n_metric);
    J.resize(num_qp);

    auto A_d1 = [&](Index i) {
        return R[1].middleRows(i * num_qp, num_qp);
    };

    for (Index q = 0; q < num_qp; ++q) {
        Eigen::Matrix<T, d, d> A_mat;
        for (Index i = 0; i < d; ++i)
            for (Index j = i; j < d; ++j) {
                // A_{αβ}   = A_α · A_β
                const T A_ij = A_d1(i).row(q).dot(A_d1(j).row(q));
                A_mat(i, j) = A_ij;
                if (j != i) A_mat(j, i) = A_ij;
            }

        // A^{αβ} = (A_αβ)^{-1}
        const Eigen::Matrix<T, d, d> inv_A = A_mat.inverse();
        // det A_{αβ}
        const T det_A = A_mat.determinant();

        for (Index i = 0; i < d; ++i)
            for (Index j = i; j < d; ++j) {
                // Allocate results into output arrays
                const Index packed = pack2<d>(i, j);
                A    (q, packed) = A_mat(i, j);
                A_inv(q, packed) = inv_A(i, j);
            }
        // J = \sqrt{ det A_{αβ} }
        J(q) = std::sqrt(det_A);
    }
}

// === Projections ====================================================================

/**
 * @brief Project an ambient-space vector field v (Q × 3) onto the covariant
 *        basis: v_α(q) = v(q) · a_α(q). Output is (Q × d).
 */
template <std::floating_point T, std::size_t d>
inline void project_to_covariant(const ColMatrix<T, 3>& v,
                                 const std::vector<ColMatrix<T, 3>>& position_data,
                                 Matrix<T>& v_cov)
{
    const Index Q = v.rows();
    v_cov.resize(Q, d);

    auto a_view = [&](Index i) {
        return position_data[1].middleRows(i * Q, Q);
    };

    for (Index q = 0; q < Q; ++q)
        for (std::size_t alpha = 0; alpha < d; ++alpha)
            v_cov(q, alpha) = v.row(q).dot(a_view(alpha).row(q));
}

/**
 * @brief Raise covariant components to contravariant via the inverse metric:
 *        v^α(q) = g^{αβ}(q) v_β(q). Output is (Q × d).
 */
template <std::floating_point T, std::size_t d>
inline void project_to_contravariant(const Matrix<T>& v_cov,
                                     const Matrix<T>& g_inv_data,
                                     Matrix<T>& v_up)
{
    const Index Q = v_cov.rows();
    v_up.resize(Q, d);

    for (Index q = 0; q < Q; ++q)
        for (std::size_t alpha = 0; alpha < d; ++alpha) {
            T sum = T(0);
            for (std::size_t beta = 0; beta < d; ++beta)
                sum += g_inv_data(q, pack2<d>(alpha, beta)) * v_cov(q, beta);
            v_up(q, alpha) = sum;
        }
}

} // namespace geometry::intrinsic

} // namespace pyck

#endif // PYCK_PRIMITIVES_INTRINSIC_HPP
