#include "bspline.hpp"

#include "bspline_algorithms.hpp"
#include "refinement_algorithms.hpp"
#include "../memory.hpp"

namespace pyck
{

// === Evaluation =====================================================================

template <std::floating_point T>
void BSpline<T>::eval_on_span(const Vector<T>& points, Index span_idx, Index order,
                              std::vector<Matrix<T>>& uni_results) const
{
    const Index Q = points.size();
    const Index N = this->degree_ + 1;

    for (Index k = 0; k <= order; ++k) uni_results[k].resize(N, Q);

    STACK_ARRAY(T, ndu_fn, N * N);
    STACK_ARRAY(T, ndu_kd, N * N);
    STACK_ARRAY(T, left,   N);
    STACK_ARRAY(T, right,  N);
    STACK_ARRAY(T, a,      2 * N);

    // Loop over quadrature points
    for (Index q = 0; q < Q; ++q) {
        // --- Cox-de Boor ------------------------------------------------------------
        // uni_results[0].col(q), N bsp values at points(q), filled in-place
        basis::eval::cox_de_boor_at<T>(this->degree_, this->knots_,
                                       points(q), span_idx, q,
                                       uni_results,
                                       ndu_fn, ndu_kd, left, right);

        // --- Derivative Recurrence --------------------------------------------------
        // uni_results[k].col(q) (k = 1 .. order), N bsp derivatives, filled in-place
        if (order > 0) {
            basis::eval::derivative_recurrence_at<T>(this->degree_, order, q,
                                                     uni_results,
                                                     ndu_fn, ndu_kd, a);
        }
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
