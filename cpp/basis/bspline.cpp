#include "bspline.hpp"

#include "evaluation.hpp"
#include "refinement.hpp"

namespace pyck
{

// === Evaluation =====================================================================

template <std::floating_point T>
void BSpline<T>::eval_on_span(const Vector<T>& points, Index span_idx,
                              std::vector<Matrix<T>>& results,
                              Evaluator<T>& eval) const
{
    const Index Q = points.size();
    const Index N = this->degree_ + 1;
    const Index order = static_cast<Index>(results.size()) - 1;

    eval.resize(N, order);
    for (Index k = 0; k <= order; ++k) results[k].resize(N, Q);

    // Q-loop over single-point P&T A2.2 / A2.3 kernels; each iteration fills
    // one contiguous column of every results[k].
    for (Index q = 0; q < Q; ++q) {
        basis::eval::cox_de_boor_at<T>(
            this->degree_, this->knots_, points(q), span_idx,
            eval.point_derivs.col(0), eval);

        if (order > 0)
            basis::eval::derivative_recurrence_at<T>(
                this->degree_, order, eval.point_derivs, eval);

        for (Index k = 0; k <= order; ++k)
            results[k].col(q) = eval.point_derivs.col(k);
    }
}

// === Refinement =====================================================================

template <std::floating_point T>
std::pair<Ptr<Basis<T>>, Matrix<T>> BSpline<T>::insert_knot(T u) const
{
    auto [kv_new, transform] = basis::refine::insert_knot(this->degree_, this->knots_, u);

    return {std::make_shared<BSpline<T>>(this->degree_, std::move(kv_new)),
            std::move(transform)};
}

template <std::floating_point T>
std::pair<Ptr<Basis<T>>, Matrix<T>> BSpline<T>::elevate_degree() const
{
    auto [kv_new, transform] = basis::refine::elevate_degree(this->degree_, this->knots_);
    
    return {std::make_shared<BSpline<T>>(this->degree_ + 1, std::move(kv_new)),
            std::move(transform)};
}

// === Factory Methods ================================================================

template <std::floating_point T>
BSpline<T> BSpline<T>::clamped_uniform(Index degree, Index num_basis)
{
    auto knots = KnotVector<T>::clamped_uniform(degree, num_basis);
    return BSpline<T>(degree, std::move(knots));
}

// === Template Instantiations ========================================================

template class BSpline<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BSpline<float>;
#endif

} // namespace pyck
