#include "quadrature.hpp"

namespace pyck
{

template <std::floating_point T>
std::pair<Vector<T>, Vector<T>> QuadratureRule<T>::map_to_domain(T lo, T hi) const
{
    std::size_t Q = num_points();
    Vector<T> mapped_pts(Q);
    Vector<T> mapped_weights(Q);

    T half_len = static_cast<T>(0.5) * (hi - lo);
    for (std::size_t q = 0; q < Q; ++q)
    {
        mapped_pts(q) = lo + half_len * (points_(q) + static_cast<T>(1));
        mapped_weights(q) = half_len * weights_(q);
    }
    return {mapped_pts, mapped_weights};
}

// === Template Instantiations ========================================================

template class pyck::QuadratureRule<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class pyck::QuadratureRule<float>;
#endif

} // namespace pyck
