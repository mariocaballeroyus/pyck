#ifndef PYCK_KNOT_VECTOR_HPP
#define PYCK_KNOT_VECTOR_HPP

#include <cstddef>
#include <vector>

namespace pyck
{

/**
 * @brief Non-decreasing knot vector for basis functions.
 *
 * @details Knot vectors are the building blocks used to define B-Spline and NURBS 
 *          basis functions. They are defined over the parametric domain, typically 
 *          in the unit interval [0, 1]. 
 *
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class KnotVector
{
public:

    // === Constructors ===============================================================

    /// @brief Default constructor. Creates an empty knot vector.
    KnotVector() = default;

    /**
     * @brief Construct a knot vector from a sequence of knot values.
     *
     * @param knots Non-decreasing sequence of knot values.
     */
    explicit KnotVector(std::vector<T> knots);

    // === Utility ====================================================================

    /**
     * @brief Find the knot span index for a given parameter value.
     *
     * @details Returns the index `i` such that `knots[i] <= point < knots[i+1]`,
     *          with the convention that the last span is closed on the right.
     *
     * @param degree Polynomial degree.
     * @param point Parameter value to locate.
     * @return Span index.
     */
    int find_span(int degree, T point) const;

    /**
     * @brief Get the parametric bounds of a knot span.
     *
     * @param span Span index in `[0, num_spans())`.
     * @return Pair `(knots[span], knots[span+1])`.
     */
    std::pair<T, T> span_bounds(int span) const { return {knots_[span], knots_[span + 1]}; }

    // === Properties =================================================================

    /// @brief Number of knots in the vector.
    int size() const { return static_cast<int>(knots_.size()); }

    /// @brief Total number of knot spans (including zero-length clamped ones).
    int num_spans() const { return static_cast<int>(knots_.size()) - 1; }

    /// @brief Access the i-th knot value.
    T operator[](int i) const { return knots_[i]; }

    /// @brief First knot value.
    T front() const { return knots_.front(); }

    /// @brief Last knot value.
    T back() const { return knots_.back(); }

    /// @brief Read-only reference to the underlying knot values.
    const std::vector<T>& data() const { return knots_; }

    // === Refinement =================================================================

    /**
     * @brief Insert a knot value into the knot vector.
     *
     * @details If `u` already exists, its multiplicity increases by 1. Otherwise, `u` is
     *          added with multiplicity 1. The non-decreasing order is preserved.
     *
     * @param u Knot value to insert. Must lie in `[front(), back()]`.
     * @return A new KnotVector with `u` inserted.
     */
    KnotVector<T> insert_knot(T u) const;

    /**
     * @brief Drop one copy of a knot value from the knot vector.
     *
     * @details If `u` has multiplicity > 1 the multiplicity decreases by 1.
     *          If multiplicity is exactly 1 the value is removed entirely.
     *
     * @param u Knot value to drop. Must be present in the knot vector.
     * @return A new KnotVector with one copy of `u` removed.
     * @throws std::invalid_argument if `u` is not present.
     */
    KnotVector<T> drop_knot(T u) const;

    /**
     * @brief Degree-elevate the knot vector once.

     * @details Each unique knot's multiplicity is incremented by 1, keeping it clamped if 
     *          the original vector was, and conserving the total number of spans
     *          (and hence the degree of the resulting basis).
     *
     * @return The degree-elevated knot vector.
     */
    KnotVector<T> elevate() const;

    // === Factory Methods =============================================================

    /**
     * @brief Create a clamped, uniformly-spaced knot vector on [0, 1].

     * @details Creates a knot vector with `(degree+1)` repeated knots at each end and
     *          uniformly spaced internal knots.

     * @param degree Polynomial degree.
     * @param num_basis Number of basis functions. Must satisfy `num_basis >= degree + 1`.
     * @return The clamped uniform knot vector.
     */
    static KnotVector<T> clamped_uniform(int degree, int num_basis);

private:

    /// @brief Non-decreasing sequence of knot values.
    std::vector<T> knots_;

}; // class KnotVector<T>

} // namespace pyck

#endif // PYCK_KNOT_VECTOR_HPP
