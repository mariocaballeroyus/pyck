#include "nurbs.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "evaluation.hpp"
#include "refinement.hpp"

namespace pyck
{

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
                            std::vector<Matrix<T>>& results,
                            Evaluator<T>& eval) const
{
    const Index p = this->degree_;
    const Index Q = points.size();
    const Index N = p + 1;
    const Index order = static_cast<Index>(results.size()) - 1;

    eval.resize(N, order);
    for (Index k = 0; k <= order; ++k) results[k].resize(N, Q);

    // The (p+1) active basis indices on this span are span_idx-p .. span_idx.
    Vector<T> active = weights_.segment(span_idx - p, N);

    // Q-loop: for each point we compute the B-spline derivs into
    // eval.point_derivs, then apply the rational quotient at that q,
    // writing R[k].col(q) directly.
    for (Index q = 0; q < Q; ++q) {
        basis::eval::cox_de_boor_at<T>(
            p, this->knots_, points(q), span_idx,
            eval.point_derivs.col(0), eval);

        if (order > 0)
            basis::eval::derivative_recurrence_at<T>(
                p, order, eval.point_derivs, eval);

        basis::eval::apply_rational_quotient_at<T>(
            eval.point_derivs, active, order, q, results);
    }
}

// === Knot Insertion =================================================================

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

    for (Index i = 0; i < n_new; ++i) {
        if (i + p <= k) {
            M(i, i)  = T(1);
            w_new(i) = w_old(i);
        } else if (i <= k) {
            const T denom = kv[i + p] - kv[i];
            const T alpha = (std::abs(denom) > eps)
                          ? (u - kv[i]) / denom : T(0);
            const T wn = alpha * w_old(i) + (T(1) - alpha) * w_old(i - 1);
            w_new(i) = wn;

            const T inv_wn = (std::abs(wn) > eps) ? T(1) / wn : T(0);
            const T beta   = alpha * w_old(i) * inv_wn;
            M(i, i - 1) = T(1) - beta;
            M(i, i)     = beta;
        } else {
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

    auto [transform, w_new] = single_nurbs_insertion<T>(p, this->knots_, weights_, u);
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
    auto [kv_new, M_bs] = basis::refine::elevate_degree(p, this->knots_);
    const Index n_new = M_bs.rows();

    // New weights: w'_i = sum_j M_bs[i,j] * w_j.
    Vector<T> w_new = M_bs * weights_;

    // Rational CP transform: M_rat[i,j] = M_bs[i,j] * w_j / w'_i.
    Matrix<T> M_rat(n_new, n_old);
    const T eps = T(1e-14);
    for (Index i = 0; i < n_new; ++i) {
        const T inv_w = (std::abs(w_new(i)) > eps) ? T(1) / w_new(i) : T(0);
        for (Index j = 0; j < n_old; ++j) {
            M_rat(i, j) = M_bs(i, j) * weights_(j) * inv_w;
        }
    }

    auto new_basis = std::make_shared<NURBS<T>>(
        p + 1, std::move(kv_new), std::move(w_new));

    return {new_basis, std::move(M_rat)};
}

// === Factory Methods ================================================================

template <std::floating_point T>
NURBS<T> NURBS<T>::clamped_uniform(Index degree, Index num_basis, const Vector<T>& weights)
{
    auto knots = KnotVector<T>::clamped_uniform(degree, num_basis);
    return NURBS<T>(degree, std::move(knots), weights);
}

// === Template Instantiations ========================================================

template class NURBS<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class NURBS<float>;
#endif


} // namespace pyck
