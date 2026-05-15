#include "knots.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pyck
{

template <std::floating_point T>
KnotVector<T>::KnotVector(Vector<T> knots)
    : knots_(std::move(knots))
{
    if (knots_.size() == 0) {
        throw std::invalid_argument("KnotVector: "
                                    "knot sequence must not be empty.");
    }
    if (!std::is_sorted(knots_.begin(), knots_.end())) {
        throw std::invalid_argument("KnotVector: "
                                    "knot sequence must be non-decreasing.");
    }
}

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

template <std::floating_point T>
KnotVector<T> KnotVector<T>::insert(T u, Index count) const
{
    if (u < knots_(0) || u > knots_(knots_.size() - 1)) {
        throw std::invalid_argument("KnotVector::insert: "
                                    "u is outside the knot range.");
    }
    if (count == 0) {
        return *this;
    }

    // Locate the insertion point (last position where knots_[pos-1] <= u).
    auto it = std::upper_bound(knots_.begin(), knots_.end(), u);
    const Index pos = static_cast<Index>(std::distance(knots_.begin(), it));

    Vector<T> new_knots(knots_.size() + count);
    new_knots.head(pos) = knots_.head(pos);
    new_knots.segment(pos, count).setConstant(u);
    new_knots.tail(knots_.size() - pos) = knots_.tail(knots_.size() - pos);

    return KnotVector<T>(std::move(new_knots));
}

template <std::floating_point T>
KnotVector<T> KnotVector<T>::elevate(Index count) const
{
    if (count == 0) {
        return *this;
    }

    // Each run of equal knots has its length (multiplicity) increased by `count`.
    std::vector<T> elevated;
    elevated.reserve(knots_.size() + count * 2);

    Index i = 0;
    const Index n = knots_.size();
    while (i < n) {
        Index j = i;
        while (j < n && knots_(j) == knots_(i)) ++j;
        const Index mult = (j - i) + count;
        for (Index k = 0; k < mult; ++k) {
            elevated.push_back(knots_(i));
        }
        i = j;
    }

    Vector<T> out(static_cast<Eigen::Index>(elevated.size()));
    for (std::size_t k = 0; k < elevated.size(); ++k) {
        out(static_cast<Eigen::Index>(k)) = elevated[k];
    }
    return KnotVector<T>(std::move(out));
}

// === Factory Methods ================================================================

template <std::floating_point T>
KnotVector<T> clamped_uniform_knots(Index degree,
                                    Index num_basis)
{
    // Check: num_knots = num_basis + degree + 1
    if (num_basis < degree + 1) {
        throw std::invalid_argument("KnotVector: "
                                    "Number of basis functions must be at least degree + 1.");
    }

    Index num_knots = num_basis + degree + 1;
    Vector<T> knots(num_knots);

    // Number of unique spans in the internal part of the vector
    const Index num_spans = num_basis - degree;

    for (Index i = 0; i < num_knots; ++i)
    {
        if (i <= degree) {
            // Clamped start: first p+1 knots are 0.0
            knots(i) = static_cast<T>(0.0);
        }
        else if (i >= num_knots - degree - 1) {
            // Clamped end: last p+1 knots are 1.0
            knots(i) = static_cast<T>(1.0);
        }
        else {
            // Uniformly spaced internal knots
            knots(i) = (static_cast<T>(i - degree) / static_cast<T>(num_spans));
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
