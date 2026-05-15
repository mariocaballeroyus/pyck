#ifndef PYCK_KNOT_VECTOR_HPP
#define PYCK_KNOT_VECTOR_HPP

#include <cstddef>

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

    // === Constructors ===============================================================

    /**
     * @brief Construct a knot vector from a sequence of knot values.
     *
     * @param knots Non-decreasing sequence of knot values.
     * @throws std::invalid_argument if the sequence is empty or not non-decreasing.
     */
    explicit KnotVector(Vector<T> knots);

    // === Utility Methods ============================================================

    /**
     * @brief Find the knot span index for a given parameter value.
     *
     * Returns the index `i` such that knots[i] <= point < knots[i+1],
     * with the convention that the last span is closed on the right.
     *
     * @param degree Polynomial degree.
     * @param point Parameter value.
     * @return Span index.
     */
    Index find_span(Index degree, T point) const;

    // === Properties =================================================================

    /**
     * @brief Get the parametric bounds of a knot span.
     *
     * @param span Span index in [0, num_spans()).
     * @return Pair (knots[span], knots[span+1]).
     */
    std::pair<T, T> span_bounds(Index span) const
    { return {knots_[span], knots_[span + 1]}; }

    /// @brief Number of basis functions of given degree supported by this vector.
    Index num_basis(Index degree) const
    { return knots_.size() - degree - 1; }

    /// @brief Number of knots in the vector.
    Index size() const
    { return knots_.size(); }

    /// @brief Total number of knot spans (including zero-length clamped ones).
    Index num_spans() const
    { return knots_.size() - 1; }

    /// @brief Access the i-th knot value.
    T operator[](Index i) const
    { return knots_[i]; }

    /// @brief First knot value.
    T front() const
    { return knots_(0); }

    /// @brief Last knot value.
    T back() const
    { return knots_(knots_.size() - 1); }

    /// @brief Read-only reference to the underlying Eigen vector.
    const Vector<T>& data() const
    { return knots_; }

    // === Knot Insertion =============================================================

    /**
     * @brief Insert a knot value @p u into the knot vector @p count times.
     *
     * @param u     Knot value to insert. Must lie in [front(), back()].
     * @param count Number of times to insert (default 1).
     * @return A new KnotVector with @p u inserted @p count times.
     */
    KnotVector<T> insert(T u, Index count = 1) const;

    // === Degree Elevation ===========================================================

    /**
     * @brief Degree-elevate the knot vector @p count times.
     *
     * Each unique knot's multiplicity is incremented by @p count. Pure
     * splice operation; the basis-level @c Basis::elevate_degree pairs
     * this with the corresponding control-point transform.
     */
    KnotVector<T> elevate(Index count = 1) const;

private:

    /// @brief Non-decreasing sequence of knot values.
    Vector<T> knots_;

};

// === Factory Methods ================================================================

/**
 * @brief Create a clamped, uniformly-spaced knot vector on [0, 1].
 * @tparam T Scalar type
 * @param degree   Polynomial degree (p).
 * @param num_basis Number of basis functions (n).  Must satisfy n >= p + 1.
 * @return A knot vector with (p+1) repeated knots at each end and
 *         uniformly spaced internal knots.
 *
 * @throws std::invalid_argument if num_basis < degree + 1 or negative.
 */
template <std::floating_point T = double>
KnotVector<T> clamped_uniform_knots(Index degree, Index num_basis);

} // namespace pyck

#endif // PYCK_KNOT_VECTOR_HPP
