#ifndef PYCK_EVALUATION_HPP
#define PYCK_EVALUATION_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "knot_vector.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Single-point scratch workspace for the Cox–de Boor recurrence.
 *        The Q-loop is driven by the caller (`BSpline::eval_on_span` /
 *        `NURBS::eval_on_span`); these buffers hold state for one point.
 */
template <std::floating_point T>
struct Evaluator
{
    /// @brief Triangular table of basis values at one point;
    ///        ndu_fn(r, j) = N_{i-j+r, j}(u). Filled by cox_de_boor_at,
    ///        read by derivative_recurrence_at. (N, N).
    Matrix<T> ndu_fn;

    /// @brief Knot-difference denominators (N^2), flat-indexed as
    ///        ndu_kd[j * N + r].
    ColArray<T,1> ndu_kd;

    /// @brief Left span lengths per level (N): left[j] = u - knots[i + 1 - j].
    Vector<T> left;

    /// @brief Right span lengths per level (N): right[j] = knots[i + j] - u.
    Vector<T> right;

    /// @brief Derivative coefficients (2, N) — P&T A2.3 alternating rows.
    RowArray<T,2> a;

    /// @brief Per-point basis-derivative buffer; point_derivs(r, k) is the
    ///        k-th derivative of basis r at the current evaluation point.
    ///        Column 0 is written by cox_de_boor_at; columns 1..order are
    ///        written by derivative_recurrence_at. (N, order+1).
    Matrix<T> point_derivs;

    /**
     * @brief Resize the scratch buffers for a basis with `N = degree + 1`
     *        functions per span and maximum derivative order `order`.
     */
    void resize(Index N, Index order) {
        ndu_fn.resize(N, N);
        ndu_kd.resize(N * N);
        left.resize(N);
        right.resize(N);
        a.resize(2, N);
        point_derivs.resize(N, order + 1);
    }

};

namespace basis::eval
{

/**
 * @brief Compute the B-Spline basis values at a single point using Cox-de-boor 
 *        recurrence.
 *
 * @note [Piegl & Tiller] The NURBS Book, Chapter 3, Algorithm A2.2.
 */
template <std::floating_point T>
void cox_de_boor_at(Index degree, const KnotVector<T>& knots,
                    T u, Index span_idx,
                    Eigen::Ref<Vector<T>> result_N,
                    Evaluator<T>& eval)
{
    const T eps = 1e-14;
    const Index N = degree + 1; // number of basis functions per span

    // ndu_fn(r, j) = N_{i-j+r, j}(u), triangular table
    // ndu_kd[j*N + r] = knot-difference denominator for level j, basis r
    auto kd = [&](Index j, Index r) -> T& { return eval.ndu_kd[j * N + r]; };

    // Level 0: N_{i,0}(u) = 1 if u \in [u_i, u_{i+1})
    eval.ndu_fn(0, 0) = T(1);

    // Levels 1 .. p: Cox-de Boor recursion
    for (Index j = 1; j <= degree; ++j) {
        eval.left (j) = u - knots[span_idx + 1 - j];
        eval.right(j) = knots[span_idx + j] - u;

        T saved = T(0);

        for (Index r = 0; r < j; ++r) {
            kd(j, r) = eval.right(r + 1) + eval.left(j - r);

            const T inv = (std::abs(kd(j, r)) > eps)
                        ? T(1) / kd(j, r)
                        : T(0);

            const T temp = eval.ndu_fn(r, j - 1) * inv;

            eval.ndu_fn(r, j) = saved + eval.right(r + 1) * temp;
            saved             = eval.left(j - r) * temp;
        }

        eval.ndu_fn(j, j) = saved;
    }

    // Load degree-p basis values into result_N
    result_N = eval.ndu_fn.col(degree);
}

/**
 * @brief Compute the B-Spline basis derivatives at a single point using the
 *        derivative recurrence relation.
 *
 * @note [Piegl & Tiller] The NURBS Book, Chapter 3, Algorithm A2.3 (pass 2).
 */
template <std::floating_point T>
void derivative_recurrence_at(Index degree, Index order,
                              Eigen::Ref<Matrix<T>> derivs_at_point,
                              Evaluator<T>& eval)
{
    const Index N = degree + 1;

    // eval.ndu_fn and eval.ndu_kd were just filled by cox_de_boor_at.
    auto kd = [&](Index j, Index r) -> T& { return eval.ndu_kd[j * N + r]; };

    // For each active basis index r, compute its k-th derivative for k = 1..order.
    for (Index r = 0; r <= degree; ++r) {
        eval.a.setZero();
        Index s1 = 0, s2 = 1;
        eval.a(0, 0) = T(1);

        for (Index k = 1; k <= order; ++k) {
            const int rk = static_cast<int>(r) - static_cast<int>(k);
            const int pk = static_cast<int>(degree) - static_cast<int>(k);

            T d = T(0);

            if (r >= k) {
                eval.a(s2, 0) = eval.a(s1, 0) / kd(pk + 1, rk);
                d += eval.a(s2, 0) * eval.ndu_fn(rk, pk);
            }

            const int j1 = (rk >= -1) ? 1 : -rk;
            const int j2 = (static_cast<int>(r) - 1 <= pk)
                         ? static_cast<int>(k) - 1
                         : static_cast<int>(degree) - static_cast<int>(r);

            for (int j = j1; j <= j2; ++j) {
                eval.a(s2, j) = (eval.a(s1, j) - eval.a(s1, j - 1))
                              / kd(pk + 1, rk + j);
                d += eval.a(s2, j) * eval.ndu_fn(rk + j, pk);
            }

            if (static_cast<int>(r) <= pk) {
                eval.a(s2, k) = -eval.a(s1, k - 1) / kd(pk + 1, r);
                d += eval.a(s2, k) * eval.ndu_fn(r, pk);
            }

            derivs_at_point(r, k) = d;
            std::swap(s1, s2);
        }
    }

    // Multiply through by p! / (p-k)!.
    T r_factor = static_cast<T>(degree);
    for (Index k = 1; k <= order; ++k) {
        derivs_at_point.col(k) *= r_factor;
        r_factor *= static_cast<T>(degree - k);
    }
}

/**
 * @brief Binomial coefficient C(n, k), computed iteratively for small n.
 */
inline std::size_t binomial(std::size_t n, std::size_t k)
{
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    std::size_t r = 1;
    for (std::size_t i = 0; i < k; ++i) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}

/**
 * @brief Apply the rational quotient at a single point to convert B-spline
 *        derivatives into rational (NURBS-style) derivatives.
 *
 * @note [Piegl & Tiller] The NURBS Book, Chapter 4, Algorithm A4.2.
 */
template <std::floating_point T>
void apply_rational_quotient_at(const Matrix<T>& bsp_derivs,
                                const Vector<T>& active_weights,
                                Index order, Index q,
                                std::vector<Matrix<T>>& results)
{
    const Index N = static_cast<Index>(active_weights.size());

    // Per-order scalar W_k = \sum_r{ w_r * N_r^{(k)}(u) }
    std::array<T, 4> W{};
    for (Index k = 0; k <= order; ++k) {
        T s = T(0);
        for (Index r = 0; r < N; ++r)
            s += active_weights(r) * bsp_derivs(r, k);
        W[k] = s;
    }
    const T inv_W0 = (std::abs(W[0]) > T(1e-30)) ? T(1) / W[0] : T(0);

    // R[0]_r = w_r * N_r * inv_W0.
    auto col0 = results[0].col(q);
    for (Index r = 0; r < N; ++r)
        col0(r) = active_weights(r) * bsp_derivs(r, 0) * inv_W0;

    // R[k]_r = ( w_r * N_r^{(k)} - \sum_{i=1..k}{ C(k,i) * W_i * R[k-i]_r } ) / W_0.
    for (Index k = 1; k <= order; ++k) {
        auto colk = results[k].col(q);
        for (Index r = 0; r < N; ++r) {
            T num = active_weights(r) * bsp_derivs(r, k);
            for (Index i = 1; i <= k; ++i) {
                const T coeff = static_cast<T>(binomial(static_cast<std::size_t>(k),
                                                        static_cast<std::size_t>(i))) * W[i];
                num -= coeff * results[k - i](r, q);
            }
            colk(r) = num * inv_W0;
        }
    }
}

} // namespace basis::eval

} // namespace pyck

#endif // PYCK_EVALUATION_HPP
