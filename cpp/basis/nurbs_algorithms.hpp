#ifndef PYCK_NURBS_ALGORITHMS_HPP
#define PYCK_NURBS_ALGORITHMS_HPP

#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../types.hpp"

namespace pyck
{

namespace basis::eval
{

/**
 * @brief Pascal-triangle binomial table generated at compile time.
 * 
 * @tparam The maximum derivative order supported.
 */
template <int MaxOrder>
constexpr auto make_binomial_table()
{
    std::array<std::array<int, MaxOrder + 1>, MaxOrder + 1> t{};
    for (int n = 0; n <= MaxOrder; ++n) {
        t[n][0] = 1;
        for (int k = 1; k <= n; ++k)
            t[n][k] = t[n - 1][k - 1] + (k < n ? t[n - 1][k] : 0);
    }
    return t;
}

/**
 * @brief In-place rational quotient via P&T A4.4.
 *
 * @details At entry, @p uni_results[k].col(q) holds the B-spline value /
 *          derivative N_r^{(k)}(u) for k = 0..order. At exit, it holds the
 *          rational counterpart R_r^{(k)}(u).
 *
 * @note [Piegl & Tiller] The NURBS Book, Chapter 4, Algorithm A4.4.
 */
template <std::floating_point T>
void apply_rational_quotient_at(const Vector<T>& active_weights,
                                Index order, Index q,
                                std::vector<Matrix<T>>& uni_results)
{
    assert(order >= 0 && order < 4);   // wders[] / coeffs[] sized for order ≤ 3
    const Index N = static_cast<Index>(active_weights.size());

    // Compile-time binomial table; BINOM[k][i] = C(k, i) for k <= MaxOrder.
    static constexpr auto BINOM = make_binomial_table<3>();

    // Cache one contiguous column pointer per order.
    // Assumes column-major storage (Eigen default): col(q) is stride-1.
    T* col[4];
    for (Index k = 0; k <= order; ++k)
        col[k] = uni_results[k].col(q).data();

    // Pre-pass: wders[k] = W^(k)(u) = Σ_r w_r · N_r^(k).
    //  Outer-r, inner-k fuses the load of w_r across all orders.
    std::array<T, 4> wders{};
    for (Index r = 0; r < N; ++r) {
        const T w_r = active_weights(r);
        for (Index k = 0; k <= order; ++k)
            wders[k] += w_r * col[k][r];
    }
    const T inv_w0 = (std::abs(wders[0]) > T(1e-30)) ? T(1) / wders[0] : T(0);

    // --- Pre-multiply binomial * wders coefficients (used by every r).
    T coeffs[4][4]{};
    for (Index k = 1; k <= order; ++k)
        for (Index i = 1; i <= k; ++i)
            coeffs[k][i] = static_cast<T>(BINOM[k][i]) * wders[i];

    std::array<T, 4> Aders{}, CK{};
    for (Index r = 0; r < N; ++r) {
        const T w_r = active_weights(r);
        for (Index k = 0; k <= order; ++k)
            Aders[k] = w_r * col[k][r];

        for (Index k = 0; k <= order; ++k) {
            T v = Aders[k];
            for (Index i = 1; i <= k; ++i)
                v -= coeffs[k][i] * CK[k - i];
            CK[k] = v * inv_w0;
        }

        for (Index k = 0; k <= order; ++k)
            col[k][r] = CK[k];
    }
}

} // namespace basis::eval

} // namespace pyck

#endif // PYCK_NURBS_ALGORITHMS_HPP
