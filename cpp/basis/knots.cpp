#include "knots.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
Index KnotVector<T>::find_span(Index degree, T param) const
{
    // Basis indexing 0 .. n
    Index num_knots = knots_.size();
    Index n = num_knots - degree - 2;

    // Edge cases
    if (param >= knots_[n + 1]) return n;
    if (param <= knots_[degree]) return degree;

    // Binary search for the span
    auto it = std::upper_bound(knots_.begin() + degree, knots_.begin() + n + 1, param);
    return static_cast<Index>(std::distance(knots_.begin(), it) - 1);
}

// === Factory Methods ================================================================

template <std::floating_point T>
KnotVector<T> clamped_uniform_knots(Index degree, 
                                    Index num_basis)
{
    // Check: num_knots = num_basis + degree + 1
    if (num_basis < degree + 1) {
        throw std::invalid_argument(
            "Number of basis functions must be at least degree + 1."
        );
    }

    Index num_knots = num_basis + degree + 1;
    std::vector<T> knots(num_knots);

    // Number of unique spans in the internal part of the vector
    const Index num_spans = num_basis - degree;

    for (Index i = 0; i < num_knots; ++i) 
    {
        if (i <= degree) {
            // Clamped start: first p+1 knots are 0.0
            knots[i] = static_cast<T>(0.0);
        }  
        else if (i >= num_knots - degree - 1) {
            // Clamped end: last p+1 knots are 1.0
            knots[i] = static_cast<T>(1.0);
        }  
        else {
            // Uniformly spaced internal knots
            knots[i] = static_cast<T>(static_cast<double>(i - degree) / static_cast<double>(num_spans));
        }
    }

    return KnotVector<T>(std::move(knots));
}

// === Template Instantiations ========================================================

template class KnotVector<double>;
template KnotVector<double> clamped_uniform_knots<double>(Index, Index);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class KnotVector<float>;
template KnotVector<float> clamped_uniform_knots<float>(Index, Index);
#endif

} // namespace pyck
