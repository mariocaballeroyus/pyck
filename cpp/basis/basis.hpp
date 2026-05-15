#ifndef PYCK_BASIS_HPP
#define PYCK_BASIS_HPP

#include <cstddef>
#include <vector>
#include <concepts>

#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Abstract base class for basis functions defined on a one-dimensional 
 *        parametric space
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class Basis
{
public:

    // === Constructors ===============================================================

    /// @brief Virtual destructor
    virtual ~Basis() = default;

    /**
     * @brief Construct a basis with the given degree and knots
     * 
     * @param degree Polynomial degree of the basis functions
     * @param knots Knot vector defining the basis functions
     */
    explicit Basis(Index degree, KnotVector<T> knots) 
    : degree_(degree), knots_(std::move(knots)) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the (p+1) non-zero basis functions and their derivatives
     *        at given parameter values within a single knot span.
     * 
     * @param points Parameter values (all assumed to lie in knot span `span`).
     * @param span   Knot-span index (as returned by KnotVector::find_span).
     * @param order  Highest order of derivatives to compute.
     * @return A vector of (order+1) matrices, each of size (m, p+1).
     *         results[k](i, j) = d^k N_{span-p+j,p} / du^k (points[i])
     */
    virtual std::vector<Matrix<T>> eval_on_span(const Vector<T>& points,
                                               Index span,
                                               Index order = 0) const = 0;

    /**
     * @brief Evaluate all basis functions and their derivatives at given parameter values.
     * 
     * @param points Parameter values.
     * @param order  Highest order of derivatives to compute (default 0).
     * @return A vector of (order+1) matrices, each of size (m, n).
     */
    virtual std::vector<Matrix<T>> eval_all(const Vector<T>& points,
                                            Index order = 0) const = 0;

    // === Properties =================================================================

    /// @brief Get the degree of the basis functions
    Index degree() const 
    { return degree_; }

    /// @brief Get the number of basis functions
    virtual Index num_basis() const 
    { return knots_.num_basis(degree_); }

    /// @brief Get the number of parametric intervals (elements) for this basis
    virtual Index num_intervals() const 
    { return knots_.num_spans(); }

    /**
     * @brief Find the knot span containing the given parameter value.
     *
     * Convenience wrapper for knot_vector().find_span(degree(), param).
     *
     * @param param Parameter value.
     * @return Span index.
     */
    Index find_span(T param) const
    { return knots_.find_span(degree_, param); }

    /**
     * @brief Compute the Greville abscissae for this basis.
     * 
     * @return A vector of parameter values corresponding to the Greville points.
     */
    virtual Vector<T> greville_abscissae() const = 0;

    /// @brief Get the knot vector object
    const KnotVector<T>& knot_vector() const 
    { return knots_; }

    /// @brief Get the raw knot values
    const Vector<T>& knots() const
    { return knots_.data(); }

protected:

    // === Member Variables ===========================================================

    /// @brief Degree of the basis functions
    Index degree_;

    /// @brief Knot vector defining the basis functions
    KnotVector<T> knots_;

};

} // namespace pyck

#endif // PYCK_BASIS_HPP
