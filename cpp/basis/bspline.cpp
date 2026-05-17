#include "bspline.hpp"

#include "evaluation.hpp"
#include "refinement.hpp"

namespace pyck
{

template <std::floating_point T>
void BSpline<T>::eval_on_span(const Vector<T>& points,
                              Index span_idx,
                              Index order,
                              std::vector<Matrix<T>>& results) const
{
    bspline_kernel::eval_on_span(this->degree_, this->knots_,
                                 points, span_idx, order, results);
}

template <std::floating_point T>
Vector<T> BSpline<T>::greville_abscissae() const
{
    return bspline_kernel::greville_abscissae(this->degree_, this->knots_);
}

template <std::floating_point T>
std::pair<Ptr<Basis<T>>, Matrix<T>> BSpline<T>::insert_knot(T u) const
{
    auto [kv_new, transform] =
        bspline_kernel::insert_knot(this->degree_, this->knots_, u);
    return {std::make_shared<BSpline<T>>(this->degree_, std::move(kv_new)),
            std::move(transform)};
}

template <std::floating_point T>
std::pair<Ptr<Basis<T>>, Matrix<T>> BSpline<T>::elevate_degree() const
{
    auto [kv_new, transform] =
        bspline_kernel::elevate_degree(this->degree_, this->knots_);
    return {std::make_shared<BSpline<T>>(this->degree_ + 1, std::move(kv_new)),
            std::move(transform)};
}

// === Template Instantiations ========================================================

template class BSpline<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BSpline<float>;
#endif

} // namespace pyck
