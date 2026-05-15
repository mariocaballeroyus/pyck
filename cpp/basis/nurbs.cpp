#include "nurbs.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pyck
{

namespace {

/// Binomial coefficient C(n, k), computed iteratively for small n.
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
 * Apply the rational quotient-rule recurrence to a stack of B-spline
 * derivative matrices.
 *
 * Inputs:
 *   bspline_derivs[k](q, j) = N_j^{(k)}(u_q) for each evaluation point u_q
 *                              and each *active* basis index j (column).
 *   active_weights[j]       = weight for the j-th active basis index.
 *
 * Output:
 *   result[k](q, j) = R_j^{(k)}(u_q) (rational shape function and derivatives).
 *
 * Recurrence (per evaluation point, per basis index j):
 *
 *   A_j^{(k)} = w_j N_j^{(k)}
 *   W^{(k)}   = Σ_j A_j^{(k)}
 *   R_j^{(k)} = ( A_j^{(k)} - Σ_{i=1..k} C(k,i) W^{(i)} R_j^{(k-i)} ) / W
 */
template <std::floating_point T>
std::vector<Matrix<T>> apply_rational_quotient(
    const std::vector<Matrix<T>>& bspline_derivs,
    const Vector<T>& active_weights)
{
    const Index order = static_cast<Index>(bspline_derivs.size()) - 1;
    const Index Q = bspline_derivs[0].rows();
    const Index K = bspline_derivs[0].cols();

    // A_k(q, j) = w_j * N_j^{(k)}(u_q)
    std::vector<Matrix<T>> A(order + 1);
    for (Index k = 0; k <= order; ++k) {
        A[k] = bspline_derivs[k];
        for (Index j = 0; j < K; ++j) {
            A[k].col(j) *= active_weights(j);
        }
    }

    // W_k(q) = sum over j of A_k(q, j)
    std::vector<Vector<T>> W(order + 1);
    for (Index k = 0; k <= order; ++k) {
        W[k] = A[k].rowwise().sum();
    }

    std::vector<Matrix<T>> R(order + 1);
    for (Index k = 0; k <= order; ++k) {
        R[k] = Matrix<T>(Q, K);
    }

    for (Index q = 0; q < Q; ++q)
    {
        const T W0 = W[0](q);
        const T inv_W0 = (std::abs(W0) > T(1e-30)) ? T(1) / W0 : T(0);

        // 0th order: R_j = A_j / W
        R[0].row(q) = A[0].row(q) * inv_W0;

        // Higher orders via the quotient recurrence
        for (Index k = 1; k <= order; ++k)
        {
            // numerator = A_k - Σ_{i=1..k} C(k, i) * W_i * R_{k-i}
            Eigen::Matrix<T, 1, Eigen::Dynamic> num = A[k].row(q);
            for (Index i = 1; i <= k; ++i) {
                const T coeff = static_cast<T>(binomial(
                    static_cast<std::size_t>(k),
                    static_cast<std::size_t>(i))) * W[i](q);
                num.noalias() -= coeff * R[k - i].row(q);
            }
            R[k].row(q) = num * inv_W0;
        }
    }

    return R;
}

} // namespace

template <std::floating_point T>
NURBS<T>::NURBS(Index degree, KnotVector<T> knots, Vector<T> weights)
    : BasisType(degree, knots),
      bspline_(degree, std::move(knots)),
      weights_(std::move(weights))
{
    const Index n = bspline_.num_basis();
    if (static_cast<Index>(weights_.size()) != n) {
        throw std::invalid_argument(
            "NURBS: number of weights (" + std::to_string(weights_.size())
            + ") must match the number of basis functions ("
            + std::to_string(n) + ").");
    }
    if ((weights_.array() <= T(0)).any()) {
        throw std::invalid_argument(
            "NURBS: all weights must be strictly positive.");
    }
}

template <std::floating_point T>
std::vector<Matrix<T>> NURBS<T>::eval_on_span(const Vector<T>& points,
                                              Index span,
                                              Index order) const
{
    auto bspline_derivs = bspline_.eval_on_span(points, span, order);

    // The (p+1) active basis indices on this span are span-p .. span.
    const Index p = this->degree_;
    Vector<T> active = weights_.segment(span - p, p + 1);

    return apply_rational_quotient<T>(bspline_derivs, active);
}

template <std::floating_point T>
std::vector<Matrix<T>> NURBS<T>::eval_all(const Vector<T>& points,
                                          Index order) const
{
    auto bspline_derivs = bspline_.eval_all(points, order);
    return apply_rational_quotient<T>(bspline_derivs, weights_);
}

template <std::floating_point T>
Vector<T> NURBS<T>::greville_abscissae() const
{
    return bspline_.greville_abscissae();
}

// === Template Instantiations ========================================================

template class NURBS<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class NURBS<float>;
#endif

} // namespace pyck
