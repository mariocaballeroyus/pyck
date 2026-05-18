#include "knots.hpp"

#include <algorithm>
#include <stdexcept>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
KnotVector<T>::KnotVector(std::vector<T> knots)
    : knots_(std::move(knots))
{
    if (knots_.empty()) {
        throw std::invalid_argument("KnotVector<T>: "
                                    "knot sequence must not be empty.");
    }
    if (!std::is_sorted(knots_.begin(), knots_.end())) {
        throw std::invalid_argument("KnotVector<T>: "
                                    "knot sequence must be non-decreasing.");
    }
}

// === Utility ========================================================================

template <std::floating_point T>
int KnotVector<T>::find_span(int degree, T point) const
{
    // Basis indexing 0 .. n
    const int num_knots = static_cast<int>(knots_.size());
    int n = num_knots - degree - 2;

    // Edge cases
    if (point >= knots_[n + 1]) 
        return n;
    if (point <= knots_[degree]) 
        return degree;

    // Binary search for the span
    auto it = std::upper_bound(knots_.begin() + degree, knots_.begin() + n + 1, point);
    return static_cast<int>(std::distance(knots_.begin(), it) - 1);
}

// === Refinement =====================================================================

template <std::floating_point T>
KnotVector<T> KnotVector<T>::insert_knot(T u) const
{
    if (u < knots_.front() || u > knots_.back()) {
        throw std::invalid_argument("KnotVector::insert_knot: "
                                    "u is outside the knot range.");
    }

    // Locate insertion point
    auto it = std::upper_bound(knots_.begin(), knots_.end(), u);

    // Build new knot vector
    std::vector<T> new_knots;
    new_knots.reserve(knots_.size() + 1);
    new_knots.insert(new_knots.end(), knots_.begin(), it);
    new_knots.push_back(u);
    new_knots.insert(new_knots.end(), it, knots_.end());

    return KnotVector<T>(std::move(new_knots));
}

template <std::floating_point T>
KnotVector<T> KnotVector<T>::drop_knot(T u) const
{
    auto it = std::find(knots_.begin(), knots_.end(), u);

    if (it == knots_.end()) {
        throw std::invalid_argument("KnotVector::drop_knot: "
                                    "u is not present in the knot vector.");
    }

    std::vector<T> new_knots;
    new_knots.reserve(knots_.size() - 1);
    new_knots.insert(new_knots.end(), knots_.begin(), it);
    new_knots.insert(new_knots.end(), std::next(it), knots_.end());

    return KnotVector<T>(std::move(new_knots));
}

template <std::floating_point T>
KnotVector<T> KnotVector<T>::elevate() const
{
    std::vector<T> new_knots;
    new_knots.reserve(knots_.size() * 2);

    const int n = static_cast<int>(knots_.size());
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && knots_[j] == knots_[i])
            ++j;
        // [i, j) is a run of equal knots
        for (int k = 0; k < j - i + 1; ++k)
            new_knots.push_back(knots_[i]);
        i = j;
    }

    return KnotVector<T>(std::move(new_knots));
}

// === Factory Methods ================================================================

template <std::floating_point T>
KnotVector<T> KnotVector<T>::clamped_uniform(int degree, int num_basis)
{
    if (num_basis < degree + 1) {
        throw std::invalid_argument("KnotVector<T>::clamped_uniform: "
                                    "Number of basis functions must be >= degree + 1.");
    }

    const int num_knots = num_basis + degree + 1;
    const int num_spans = num_basis - degree;
    std::vector<T> knots(num_knots);

    for (int i = 0; i < num_knots; ++i) {
        if (i <= degree)
            knots[i] = static_cast<T>(0.0);
        else if (i >= num_knots - degree - 1)
            knots[i] = static_cast<T>(1.0);
        else
            knots[i] = static_cast<T>(i - degree) / static_cast<T>(num_spans);
    }

    return KnotVector<T>(std::move(knots));
}

// === Template Instantiations ========================================================

template class KnotVector<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class KnotVector<float>;
#endif

} // namespace pyck
