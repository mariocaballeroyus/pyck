#ifndef PYCK_REFINEMENT_ALGORITHMS_HPP
#define PYCK_REFINEMENT_ALGORITHMS_HPP

#include <concepts>
#include <utility>

#include "knot_vector.hpp"
#include "../types.hpp"

namespace pyck::basis 
{

namespace refine
{

/**
 * @brief Insert a single copy of `u` into the knot vector.
 *
 * @note [Piegl & Tiller] The NURBS Book, §5.2, Algorithm A5.1 (Boehm's algorithm).
 *
 * @param degree Polynomial degree of the basis.
 * @param knots  Knot vector before insertion.
 * @param u      Knot value to insert. Must lie in [knots.front(), knots.back()].
 * @return A pair of (refined knot vector, control-point transform matrix).
 *         The transform maps old CPs to new ones: new_cps = transform * old_cps.
 */
template <std::floating_point T>
std::pair<KnotVector<T>, Matrix<T>>
insert_knot(Index degree, const KnotVector<T>& knots, T u);

/**
 * @brief Elevate the polynomial degree of the basis by one (p-refinement).
 *
 * @note [Piegl & Tiller] The NURBS Book, §5.5, Algorithm A5.9
 *       (Bezier-extraction / per-segment elevation / knot-removal composition).
 *
 * @param degree Polynomial degree of the basis before elevation.
 * @param knots  Knot vector before elevation.
 * @return A pair of (elevated knot vector, control-point transform matrix).
 *         The elevated basis has degree (degree + 1).
 */
template <std::floating_point T>
std::pair<KnotVector<T>, Matrix<T>>
elevate_degree(Index degree, const KnotVector<T>& knots);

} // namespace refine

} // namespace pyck::basis

#endif // PYCK_REFINEMENT_ALGORITHMS_HPP
