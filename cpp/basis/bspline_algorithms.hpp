#ifndef PYCK_BSPLINE_ALGORITHMS_HPP
#define PYCK_BSPLINE_ALGORITHMS_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "knot_vector.hpp"
#include "../types.hpp"

namespace pyck
{

namespace basis::eval
{

/**
 * @brief Compute the B-Spline basis values at a single point using the
 *        Cox-de-Boor recurrence.
 *
 * @details Scratch buffers are caller-owned. @p ndu_fn and @p ndu_kd are flat 
 *          col-major (N × N), @p left and @p right are length-N.
 *
 * @note [Piegl & Tiller] The NURBS Book, Chapter 3, Algorithm A2.2.
 */
template <std::floating_point T>
void cox_de_boor_at(Index degree, const KnotVector<T>& knots,
                    T u, Index span_idx, Index q,
                    std::vector<Matrix<T>>& uni_results,
                    T* ndu_fn, T* ndu_kd, T* left, T* right)
{
    const T eps = 1e-14;
    const Index N = degree + 1; // number of basis functions per span

    // Level 0: N_{i,0}(u) = 1 if u \in [u_i, u_{i+1})
    ndu_fn[0] = T(1);

    // Levels 1 .. p: Cox-de Boor recursion
    for (Index j = 1; j <= degree; ++j) {
        left[j]  = u - knots[span_idx + 1 - j];
        right[j] = knots[span_idx + j] - u;

        T saved = T(0);

        for (Index r = 0; r < j; ++r) {
            ndu_kd[j * N + r] = right[r + 1] + left[j - r];

            const T inv = (std::abs(ndu_kd[j * N + r]) > eps)
                        ? T(1) / ndu_kd[j * N + r]
                        : T(0);

            const T temp = ndu_fn[(j - 1) * N + r] * inv;

            ndu_fn[j * N + r] = saved + right[r + 1] * temp;
            saved             = left[j - r] * temp;
        }

        ndu_fn[j * N + j] = saved;
    }

    // Load degree-p basis values into uni_results[0].col(q).
    std::copy_n(ndu_fn + degree * N, N, uni_results[0].col(q).data());
}

/**
 * @brief Compute the B-Spline basis derivatives at a single point using the
 *        derivative recurrence relation.
 * 
 * @details Scratch buffers are caller-owned. @p ndu_fn and @p ndu_kd are flat 
 *          col-major (N × N), @p a is a flat row-major (2 x N).
 *
 * @note [Piegl & Tiller] The NURBS Book, Chapter 3, Algorithm A2.3.
 */
template <std::floating_point T>
void derivative_recurrence_at(Index degree, Index order, Index q,
                              std::vector<Matrix<T>>& uni_results,
                              const T* ndu_fn, const T* ndu_kd, T* a)
{
    const Index N = degree + 1;

    for (Index r = 0; r <= degree; ++r) {
        std::fill(a, a + 2 * N, T(0));
        Index s1 = 0, s2 = 1;
        a[0 * N + 0] = T(1);

        for (Index k = 1; k <= order; ++k) {
            const int rk = static_cast<int>(r) - static_cast<int>(k);
            const int pk = static_cast<int>(degree) - static_cast<int>(k);

            T d = T(0);

            if (r >= k) {
                a[s2 * N + 0] = a[s1 * N + 0] / ndu_kd[(pk + 1) * N + rk];
                d += a[s2 * N + 0] * ndu_fn[pk * N + rk];
            }

            const int j1 = (rk >= -1) ? 1 : -rk;
            const int j2 = (static_cast<int>(r) - 1 <= pk)
                         ? static_cast<int>(k) - 1
                         : static_cast<int>(degree) - static_cast<int>(r);

            for (int j = j1; j <= j2; ++j) {
                a[s2 * N + j] = (a[s1 * N + j] - a[s1 * N + (j - 1)])
                              / ndu_kd[(pk + 1) * N + (rk + j)];
                d += a[s2 * N + j] * ndu_fn[pk * N + (rk + j)];
            }

            if (static_cast<int>(r) <= pk) {
                a[s2 * N + k] = -a[s1 * N + (k - 1)] / ndu_kd[(pk + 1) * N + r];
                d += a[s2 * N + k] * ndu_fn[pk * N + r];
            }

            uni_results[k](r, q) = d;
            std::swap(s1, s2);
        }
    }

    T r_factor = static_cast<T>(degree);
    for (Index k = 1; k <= order; ++k) {
        uni_results[k].col(q) *= r_factor;
        r_factor *= static_cast<T>(degree - k);
    }
}

} // namespace basis::eval

} // namespace pyck

#endif // PYCK_BSPLINE_ALGORITHMS_HPP
