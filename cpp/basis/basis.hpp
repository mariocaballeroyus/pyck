#ifndef PYCK_BASIS_HPP
#define PYCK_BASIS_HPP

#include <cstddef>
#include <vector>
#include <concepts>

#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T> class Basis;

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
        : degree_(degree),
          knots_(std::move(knots)) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the non-zero (p+1) basis functions and their derivatives
     *        at given parameter values within a single knot span.
     *
     * @param points   Parameter values (all assumed to lie in the same knot span).
     * @param span_idx Knot-span index (as returned by KnotVector::find_span).
     * @param order    Maximum derivative order to compute.
     * @param results  Output buffer; the function resizes it to (order+1) matrices,
     *                 each of size (m, p+1). results[k](i, j) = d^k N_{span_idx-p+j,p}
     *                 / du^k (points[i]).
     */
    virtual void eval_on_span(const Vector<T>& points,
                              Index span_idx,
                              Index order,
                              std::vector<Matrix<T>>& results) const = 0;

    // === Utility Methods ============================================================

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

    // === Refinement =================================================================

    /**
     * @brief Refine the basis by inserting a single copy of knot value `u`.
     *
     * @param u Knot value to insert. Must lie in [knots.front(), knots.back()].
     * @return A pair of (refined basis, control-point transform matrix).
     *         The transform maps old CPs to new ones: new_cps = transform * old_cps.
     *         Callers wanting multiplicity > 1 apply the method repeatedly.
     */
    virtual std::pair<Ptr<Basis<T>>, Matrix<T>> insert_knot(T u) const = 0;

    /**
     * @brief Refine the basis by elevating the polynomial degree by one.
     *
     * Continuity at every existing internal knot is preserved (the defining
     * property of p-refinement): each unique knot's multiplicity is
     * incremented by one. Combine with `insert_knot` (elevate first, then
     * insert) for k-refinement: maximally smooth refinement at the new knots.
     *
     * @return A pair of (elevated basis, control-point transform matrix) with
     *         the same semantics as `insert_knot`. Callers wanting to elevate
     *         by more than one apply the method repeatedly.
     */
    virtual std::pair<Ptr<Basis<T>>, Matrix<T>> elevate_degree() const = 0;

    // === Properties =================================================================

    /// @brief Get the knot vector object
    const KnotVector<T>& knot_vector() const 
    { return knots_; }

    /// @brief Get the raw knot values
    const std::vector<T>& knots() const
    { return knots_.data(); }

    /// @brief Get the degree of the basis functions
    Index degree() const 
    { return degree_; }

    /// @brief Get the number of basis functions
    virtual Index num_basis() const
    { return knots_.size() - degree_ - 1; }

    /// @brief Get the number of parametric intervals (elements) for this basis
    virtual Index num_intervals() const 
    { return knots_.num_spans(); }

protected:

    /// @brief Degree of the basis functions
    Index degree_;

    /// @brief Knot vector defining the basis functions
    KnotVector<T> knots_;

};

} // namespace pyck

#endif // PYCK_BASIS_HPP
