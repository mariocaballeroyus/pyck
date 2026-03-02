#ifndef PYCK_KNOT_VECTOR_HPP
#define PYCK_KNOT_VECTOR_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "../types.hpp"

namespace pyck
{

/**
 * @brief Non-decreasing knot vector for basis functions.
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class KnotVector
{   
public:

    // === Constructors & Factory Methods =============================================

    /**
     * @brief Construct a knot vector from a sequence of knot values.
     * @param knots Non-decreasing sequence of knot values.
     */
    explicit KnotVector(std::vector<T> knots)
        : knots_(std::move(knots)) {}

    // === Properties =================================================================

    /// @brief Number of basis functions of given degree supported by this vector.
    Index num_basis(Index degree) const 
    { return knots_.size() - degree - 1; }

    /// @brief Number of knots in the vector.
    Index size() const 
    { return knots_.size(); }

    /// @brief Access the i-th knot value.
    T operator[](Index i) const 
    { return knots_[i]; }

    /// @brief First knot value.
    T front() const 
    { return knots_.front(); }

    /// @brief Last knot value.
    T back() const 
    { return knots_.back(); }

    /// @brief Read-only reference to the underlying std::vector.
    const std::vector<T>& data() const 
    { return knots_; }

    // === Iterators ==================================================================

    /// @brief Iterator to the first knot.
    auto begin() const 
    { return knots_.begin(); }

    /// @brief Iterator to the past-the-end knot.
    auto end() const 
    { return knots_.end(); }

    // === Span Methods ===============================================================

    /// @brief Total number of knot spans (including zero-length clamped ones).
    Index num_spans() const
    { return knots_.size() - 1; }

    /**
     * @brief Get the parametric bounds of a knot span.
     *
     * @param span Span index in [0, num_spans()).
     * @return Pair (knots[span], knots[span+1]).
     */
    std::pair<T, T> span_bounds(Index span) const
    { return {knots_[span], knots_[span + 1]}; }

    /**
     * @brief Find the knot span index for a given parameter value.
     *
     * Returns the index *i* such that knots[i] <= param < knots[i+1],
     * with the convention that the last span is closed on the right.
     *
     * @param degree Polynomial degree.
     * @param param Parameter value.
     * @return Span index.
     */
    Index find_span(Index degree, T param) const;

private:

    // === Member Variables ===========================================================

    /// @brief Non-decreasing sequence of knot values.
    std::vector<T> knots_;

};

// === Factory Methods ================================================================

/**
 * @brief Create a clamped, uniformly-spaced knot vector on [0, 1].
 * @tparam T Scalar type
 * @param degree   Polynomial degree (p).
 * @param num_basis Number of basis functions (n).  Must satisfy n >= p + 1.
 * @return A knot vector with (p+1) repeated knots at each end and
 *         uniformly spaced internal knots.
 */
template <std::floating_point T = double>
KnotVector<T> clamped_uniform_knots(Index degree, Index num_basis);

} // namespace pyck

#endif // PYCK_KNOT_VECTOR_HPP
