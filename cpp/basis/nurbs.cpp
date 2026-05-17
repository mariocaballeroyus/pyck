#include "nurbs.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "evaluation.hpp"
#include "refinement.hpp"

namespace pyck
{

namespace 
{

/**
 * Binomial coefficient C(n, k), computed iteratively for small n.
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
 * Apply the rational quotient-rule recurrence to a stack of B-spline
 * derivative matrices.
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
    : Basis<T>(degree, std::move(knots)),
      weights_(std::move(weights))
{
    const Index n = this->num_basis();
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
void NURBS<T>::eval_on_span(const Vector<T>& points,
                            Index span_idx,
                            Index order,
                            std::vector<Matrix<T>>& results) const
{
    std::vector<Matrix<T>> bspline_derivs;
    bspline_kernel::eval_on_span(this->degree_, this->knots_,
                                 points, span_idx, order, bspline_derivs);

    // The (p+1) active basis indices on this span are span_idx-p .. span_idx.
    const Index p = this->degree_;
    Vector<T> active = weights_.segment(span_idx - p, p + 1);

    results = apply_rational_quotient<T>(bspline_derivs, active);
}

template <std::floating_point T>
Vector<T> NURBS<T>::greville_abscissae() const
{
    return bspline_kernel::greville_abscissae(this->degree_, this->knots_);
}

// === Knot Insertion =================================================================

/**
 * One step of rational knot insertion: returns the transform mapping old
 * unweighted CPs to refined unweighted CPs, and the refined weight vector.
 */
template <std::floating_point T>
static std::pair<Matrix<T>, Vector<T>>
single_nurbs_insertion(Index p,
                       const KnotVector<T>& kv,
                       const Vector<T>& w_old,
                       T u)
{
    const T eps = T(1e-14);
    const Index n_old = kv.size() - p - 1;
    const Index n_new = n_old + 1;
    const Index k = kv.find_span(p, u);

    Matrix<T> M = Matrix<T>::Zero(n_new, n_old);
    Vector<T> w_new(n_new);

    for (Index i = 0; i < n_new; ++i)
    {
        if (i + p <= k)
        {
            M(i, i)  = T(1);
            w_new(i) = w_old(i);
        }
        else if (i <= k)
        {
            const T denom = kv[i + p] - kv[i];
            const T alpha = (std::abs(denom) > eps)
                          ? (u - kv[i]) / denom : T(0);
            const T wn = alpha * w_old(i) + (T(1) - alpha) * w_old(i - 1);
            w_new(i) = wn;

            const T inv_wn = (std::abs(wn) > eps) ? T(1) / wn : T(0);
            const T beta   = alpha * w_old(i) * inv_wn;
            M(i, i - 1) = T(1) - beta;
            M(i, i)     = beta;
        }
        else
        {
            M(i, i - 1) = T(1);
            w_new(i)    = w_old(i - 1);
        }
    }

    return {std::move(M), std::move(w_new)};
}

template <std::floating_point T>
std::pair<Ptr<Basis<T>>, Matrix<T>> NURBS<T>::insert_knot(T u) const
{
    const Index p = this->degree_;

    auto [transform, w_new] =
        single_nurbs_insertion<T>(p, this->knots_, weights_, u);
    KnotVector<T> kv_new = this->knots_.insert_knot(u);

    return {std::make_shared<NURBS<T>>(p, std::move(kv_new), std::move(w_new)),
            std::move(transform)};
}

// === Degree Elevation ===============================================================

template <std::floating_point T>
std::pair<Ptr<Basis<T>>, Matrix<T>> NURBS<T>::elevate_degree() const
{
    const Index p = this->degree_;
    const Index n_old = this->num_basis();

    // Polynomial elevation on homogeneous CPs, followed by a rational projection.
    auto [kv_new, M_bs] =
        bspline_kernel::elevate_degree(p, this->knots_);
    const Index n_new = M_bs.rows();

    // New weights: w'_i = sum_j M_bs[i,j] * w_j.
    Vector<T> w_new = M_bs * weights_;

    // Rational CP transform: M_rat[i,j] = M_bs[i,j] * w_j / w'_i.
    Matrix<T> M_rat(n_new, n_old);
    const T eps = T(1e-14);
    for (Index i = 0; i < n_new; ++i)
    {
        const T inv_w = (std::abs(w_new(i)) > eps) ? T(1) / w_new(i) : T(0);
        for (Index j = 0; j < n_old; ++j)
        {
            M_rat(i, j) = M_bs(i, j) * weights_(j) * inv_w;
        }
    }

    auto new_basis = std::make_shared<NURBS<T>>(
        p + 1, std::move(kv_new), std::move(w_new));

    return {new_basis, std::move(M_rat)};
}

// === Template Instantiations ========================================================

template class NURBS<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class NURBS<float>;
#endif

} // namespace pyck
