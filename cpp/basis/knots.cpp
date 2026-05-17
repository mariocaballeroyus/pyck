#include "knots.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
KnotVector<T>::KnotVector(Vector<T> knots)
    : knots_(std::move(knots))
{
    if (knots_.size() == 0) {
        throw std::invalid_argument("KnotVector<T>: "
                                    "knot sequence must not be empty.");
    }
    if (!std::is_sorted(knots_.begin(), knots_.end())) {
        throw std::invalid_argument("KnotVector<T>: "
                                    "knot sequence must be non-decreasing.");
    }
}

// === Utility Methods ================================================================

template <std::floating_point T>
Index KnotVector<T>::find_span(Index degree, T point) const
{
    // Basis indexing 0 .. n
    Index num_knots = knots_.size();
    Index n = num_knots - degree - 2;

    // Edge cases
    if (point >= knots_[n + 1]) return n;
    if (point <= knots_[degree]) return degree;

    // Binary search for the span
    auto it = std::upper_bound(knots_.begin() + degree, knots_.begin() + n + 1, point);
    return static_cast<Index>(std::distance(knots_.begin(), it) - 1);
}

// === Refinement =====================================================================

template <std::floating_point T>
KnotVector<T> KnotVector<T>::insert_knot(T u) const
{
    if (u < knots_(0) || u > knots_(knots_.size() - 1))
    {
        throw std::invalid_argument("KnotVector::insert_knot: "
                                    "u is outside the knot range.");
    }

    // Initialize new knot vector
    Vector<T> new_knots(knots_.size() + 1);

    // Locate insertion point
    auto it = std::upper_bound(knots_.begin(), knots_.end(), u);
    const Index pos = static_cast<Index>(std::distance(knots_.begin(), it));

    // Build new knot vector
    new_knots.head(pos) = knots_.head(pos);
    new_knots(pos) = u;
    new_knots.tail(knots_.size() - pos) = knots_.tail(knots_.size() - pos);

    return KnotVector<T>(std::move(new_knots));
}

template <std::floating_point T>
KnotVector<T> KnotVector<T>::drop_knot(T u) const
{
    auto it = std::find(knots_.begin(), knots_.end(), u);
    if (it == knots_.end())
    {
        throw std::invalid_argument("KnotVector::drop_knot: "
                                    "u is not present in the knot vector.");
    }

    const Index pos = static_cast<Index>(std::distance(knots_.begin(), it));
    const Index n_after = knots_.size() - 1;

    Vector<T> new_knots(n_after);
    new_knots.head(pos) = knots_.head(pos);
    new_knots.tail(n_after - pos) = knots_.tail(n_after - pos);

    return KnotVector<T>(std::move(new_knots));
}

template <std::floating_point T>
KnotVector<T> KnotVector<T>::elevate() const
{
    // Initialize new knot vector
    std::vector<T> new_knots;
    new_knots.reserve(knots_.size() + 2);

    // Loop over unique knots
    for (Index i = 0; i < knots_.size(); ++i)
    {
        // Find runs of equal knots
        Index j = i + 1;
        while (j < knots_.size() && knots_(j) == knots_(i))
            j++;

        // Increase multiplicity by 1
        const Index mult = j - i + 1;

        // Add knot value
        for (Index k = 0; k < mult; ++k)
            new_knots.push_back(knots_(i));

        // Move to the next unique knot
        i = j - 1;
    }

    // Convert to Eigen vector
    Vector<T> out(static_cast<Eigen::Index>(new_knots.size()));
    for (std::size_t k = 0; k < new_knots.size(); ++k)
    {
        out(static_cast<Eigen::Index>(k)) = new_knots[k];
    }
    return KnotVector<T>(std::move(out));
}

// === Factory Methods ================================================================

template <std::floating_point T>
KnotVector<T> KnotVector<T>::clamped_uniform(Index degree, Index num_basis)
{
    if (num_basis < degree + 1) 
    {
        throw std::invalid_argument("KnotVector<T>::clamped_uniform: "
                                    "Number of basis functions must be at least degree + 1.");
    }

    // Initialize vector (m = n + p + 1)
    const Index num_knots = num_basis + degree + 1;
    const Index num_spans = num_basis - degree;
    Vector<T> knots(num_knots);

    // Loop over knots and set values
    for (Index i = 0; i < num_knots; ++i)
    {
        if (i <= degree) 
        {
            // Clamped start: first p+1 knots are 0.0
            knots(i) = static_cast<T>(0.0);
        }
        else if (i >= num_knots - degree - 1) 
        {
            // Clamped end: last p+1 knots are 1.0
            knots(i) = static_cast<T>(1.0);
        }
        else 
        {
            // Uniformly spaced internal knots
            knots(i) = (static_cast<T>(i - degree) / static_cast<T>(num_spans));
        }
    }

    return KnotVector<T>(std::move(knots));
}

// === Template Instantiations ========================================================

template class KnotVector<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class KnotVector<float>;
#endif

} // namespace pyck
