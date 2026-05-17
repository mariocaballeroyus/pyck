#ifndef PYCK_EVALUATION_HPP
#define PYCK_EVALUATION_HPP

#include <concepts>
#include <vector>

#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

namespace bspline_kernel
{

/**
 * @brief Evaluate the (p+1) non-zero B-spline basis functions and their
 *        derivatives at given parameter values within a single knot span.
 *
 * @details The first pass builds a triangular table of basis function values via
 *          the Cox-de Boor recurrence. The second pass computes the derivatives
 *          via de Boor's derivative formula.
 *
 * @note [Piegl & Tiller] The NURBS Book, Chapter 3, Algorithms A2.2 and A2.3.
 *
 * @param degree   Polynomial degree of the basis.
 * @param knots    Knot vector defining the basis.
 * @param points   Parameter values (all assumed to lie in knot span `span_idx`).
 * @param span_idx Knot-span index (as returned by KnotVector::find_span).
 * @param order    Maximum derivative order to compute.
 * @param results  Output buffer; the function resizes it to (order+1) matrices,
 *                 each of size (m, p+1).
 */
template <std::floating_point T>
void eval_on_span(Index degree,
                  const KnotVector<T>& knots,
                  const Vector<T>& points,
                  Index span_idx,
                  Index order,
                  std::vector<Matrix<T>>& results);

/**
 * @brief Compute the Greville abscissae for a B-spline of the given degree
 *        and knot vector.
 *
 * @details The Greville abscissa for basis function N_{i,p} is the average
 *          of its p interior knots. They satisfy linear precision and serve
 *          as natural node points for interpolation and collocation.
 *
 * @param degree Polynomial degree of the basis.
 * @param knots  Knot vector defining the basis.
 * @returns The Greville abscissae for the basis.
 */
template <std::floating_point T>
Vector<T> greville_abscissae(Index degree, const KnotVector<T>& knots);

} // namespace bspline_kernel

} // namespace pyck

#endif // PYCK_EVALUATION_HPP
