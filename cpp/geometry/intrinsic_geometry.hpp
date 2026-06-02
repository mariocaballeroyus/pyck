#ifndef PYCK_INTRINSIC_GEOMETRY_HPP
#define PYCK_INTRINSIC_GEOMETRY_HPP

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

// === Position derivatives ===========================================================

/**
 * @brief Fill per-order packed position-derivative buffers from a basis and
 *        the active control points, up to derivative order @p max_order
 *        (clamped to the basis depth `basis.size() - 1`). Lower @p max_order
 *        trims the accumulation when a consumer needs fewer orders than the
 *        basis was evaluated to.
 */
template <std::floating_point T, std::size_t d>
inline void compute_position_derivatives(const std::vector<Matrix<T>>& basis,
                                         const ColMatrix<T, 3>& act_pts,
                                         std::vector<ColMatrix<T, 3>>& position_data,
                                         Index max_order)
{
    const Index avail = static_cast<Index>(basis.size()) - 1;
    const Index order = (max_order < avail) ? max_order : avail;
    const Index Q_    = basis[0].cols();
    const Index N_    = basis[0].rows();

    position_data.resize(order + 1);
    for (Index k = 0; k <= order; ++k) {
        const Index n_k = num_multi_indices<d>(k);
        position_data[k].resize(Q_ * n_k, 3);
    }

    for (Index q = 0; q < Q_; ++q) {
        for (Index k = 0; k <= order; ++k) {
            const Index n_k = num_multi_indices<d>(k);
            auto slab = basis[k].col(q);
            for (Index packed = 0; packed < n_k; ++packed) {
                Eigen::Matrix<T, 1, 3> deriv_q = Eigen::Matrix<T, 1, 3>::Zero();
                for (Index b = 0; b < N_; ++b) {
                    deriv_q.noalias() += slab(b * n_k + packed) * act_pts.row(b);
                }
                position_data[k].row(packed * Q_ + q) = deriv_q;
            }
        }
    }
}

// === Metric =========================================================================

/**
 * @brief Compute covariant + contravariant metric and Jacobian from position
 *        derivatives at order 1 (covariant tangents a_α).
 */
template <std::floating_point T, std::size_t d>
inline void compute_metric(const std::vector<ColMatrix<T, 3>>& position_data,
                           Matrix<T>& g_data,
                           Matrix<T>& g_inv_data,
                           Vector<T>& jac)
{
    constexpr Index n_metric = d * (d + 1) / 2;
    const Index Q_ = position_data[1].rows() / static_cast<Index>(d);

    g_data.resize(Q_, n_metric);
    g_inv_data.resize(Q_, n_metric);
    jac.resize(Q_);

    auto a_view = [&](Index i) {
        return position_data[1].middleRows(i * Q_, Q_);
    };

    for (Index q = 0; q < Q_; ++q)
    {
        Eigen::Matrix<T, d, d> g_mat;
        for (Index i = 0; i < d; ++i)
            for (Index j = i; j < d; ++j) {
                const T g_ij = a_view(i).row(q).dot(a_view(j).row(q));
                g_mat(i, j) = g_ij;
                if (j != i) g_mat(j, i) = g_ij;
            }
        const Eigen::Matrix<T, d, d> inv_g = g_mat.inverse();
        const T det_g = g_mat.determinant();
        for (Index i = 0; i < d; ++i)
            for (Index j = i; j < d; ++j) {
                const Index packed = pack2<d>(i, j);
                g_data    (q, packed) = g_mat(i, j);
                g_inv_data(q, packed) = inv_g(i, j);
            }
        jac(q) = std::sqrt(det_g);
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

#endif // PYCK_INTRINSIC_GEOMETRY_HPP
