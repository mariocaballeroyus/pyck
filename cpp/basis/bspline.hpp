#ifndef PYCK_BSPLINE_HPP
#define PYCK_BSPLINE_HPP

#include <cstddef>
#include <vector>

#include "basis.hpp"
#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief B-spline basis functions defined on a one-dimensional parametric space
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class BSpline : public Basis<T>
{
public:

    // === Constructors ===============================================================

    /// @brief Default constructor
    BSpline() = default;

    /**
     * @brief Construct a B-spline basis with the given degree and knot vector
     * 
     * @param degree Degree of the B-spline basis functions
     * @param knots Knot vector defining the B-spline basis functions
     */
    BSpline(Index degree, KnotVector<T> knots)
        : Basis<T>(degree, std::move(knots)) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the (p+1) non-zero B-spline basis functions and their
     *        derivatives at given parameter values within a single knot span.
     *
     * @details The first pass builds a triangular table of basis function values via 
     *          the Cox-de Boor recurrence. The second pass computes the derivatives 
     *          via de Boor's derivative formula.

     * @note References: 
     *       - [Piegl & Tiller] The NURBS Book, Chapter 3, Algorithms A2.2 and A2.3.
     *
     * @param points   Parameter values (all assumed to lie in knot span `span_idx`).
     * @param span_idx Knot-span index (as returned by KnotVector::find_span).
     * @param order    Maximum derivative order to compute.
     * @param results  Output buffer; the function resizes it to (order+1) matrices,
     *                 each of size (m, p+1).
     */
    void eval_on_span(const Vector<T>& points,
                      Index span_idx,
                      Index order,
                      std::vector<Matrix<T>>& results) const override;

    /**
     * @brief Compute the Greville abscissae for the B-spline basis.
     *
     * @details The Greville abscissa for basis function N_{i,p} is the average
     *          of its p interior knots. They satisfy linear precision and serve
     *          as natural node points for interpolation and collocation.
     *          
     * @returns The Greville abscissae for the B-spline basis.
     */
    Vector<T> greville_abscissae() const override;

    // === Refinement =================================================================

    /**
     * @brief Boehm knot insertion (single step).
     *
     * @param u Knot value to insert.
     * @return A pair of (refined basis, transform matrix).
     */
    std::pair<Ptr<Basis<T>>, Matrix<T>> insert_knot(T u) const override;

    /**
     * @brief Degree elevation by one (p-refinement).
     *
     * Implements the textbook Bezier-extraction / Bezier-elevation /
     * knot-removal composition (Piegl-Tiller §5.5). Continuity at each
     * existing internal knot is preserved.
     *
     * @return A pair of (elevated basis, transform matrix).
     */
    std::pair<Ptr<Basis<T>>, Matrix<T>> elevate_degree() const override;

    // === Factory Methods =============================================================

    /**
     * @brief Create a clamped, uniformly-spaced B-spline basis on [0, 1].
     * @param degree Polynomial degree.
     * @param num_basis Number of basis functions. Must satisfy `num_basis >= degree + 1`.
     * @return A B-spline basis with clamped uniform knots.
     */
    static BSpline<T> clamped_uniform(Index degree, Index num_basis)
    {
        return BSpline<T>(degree, KnotVector<T>::clamped_uniform(degree, num_basis));
    }

}; // class BSpline : public Basis<T>

} // namespace pyck

#endif // PYCK_BSPLINE_HPP
