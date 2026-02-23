#ifndef PYCK_KNOT_VECTOR_HPP
#define PYCK_KNOT_VECTOR_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace pyck
{

/**
 * @brief Non-decreasing knot vector for B-spline basis functions.
 *
 * A KnotVector is the fundamental one-dimensional data structure used to
 * define B-spline basis functions.  It stores a non-decreasing sequence
 * of knot values and provides span-finding and convenience accessors.
 */
class KnotVector
{
public:
    /**
     * @brief Construct a knot vector from a sequence of knot values.
     * @param knots Non-decreasing sequence of knot values.
     */
    KnotVector(std::vector<double> knots)
        : knots_(std::move(knots)) {}

    /**
     * @brief Create a clamped, uniformly-spaced knot vector on [0, 1].
     * @param degree   Polynomial degree (p).
     * @param num_basis Number of basis functions (n).  Must satisfy n >= p + 1.
     * @return A knot vector with (p+1) repeated knots at each end and
     *         uniformly spaced internal knots.
     */
    static KnotVector clamped_uniform(std::size_t degree, std::size_t num_basis);

    // --- Element access -------------------------------------------------------

    /// @brief Number of knots in the vector.
    std::size_t size() const { return knots_.size(); }

    /// @brief Access the i-th knot value.
    double operator[](std::size_t i) const { return knots_[i]; }

    /// @brief First knot value.
    double front() const { return knots_.front(); }

    /// @brief Last knot value.
    double back() const { return knots_.back(); }

    /// @brief Read-only reference to the underlying std::vector.
    const std::vector<double>& data() const { return knots_; }

    // --- Iterators -----------------------------------------------------------

    auto begin() const { return knots_.begin(); }
    auto end()   const { return knots_.end(); }

    // --- B-spline/NURBS utilities --------------------------------------------

    /**
     * @brief Number of basis functions of given degree supported by this vector.
     * @param degree Polynomial degree.
     * @return size() - degree - 1.
     */
    std::size_t num_basis(std::size_t degree) const { return knots_.size() - degree - 1; }

    /**
     * @brief Find the knot span index for a given parameter value.
     *
     * Returns the index *i* such that knots[i] <= u < knots[i+1],
     * with the convention that the last span is closed on the right.
     *
     * @param degree Polynomial degree.
     * @param u      Parameter value.
     * @return Span index.
     */
    std::size_t find_span(std::size_t degree, double u) const;

private:
    /// @brief Non-decreasing sequence of knot values.
    std::vector<double> knots_;
};

} // namespace pyck

#endif // PYCK_KNOT_VECTOR_HPP
