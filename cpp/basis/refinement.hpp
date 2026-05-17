#ifndef PYCK_REFINEMENT_HPP
#define PYCK_REFINEMENT_HPP

#include <concepts>
#include <utility>

#include "knots.hpp"
#include "../types.hpp"

namespace pyck::bspline_kernel
{

/**
 * @brief Insert a single copy of `u` into the knot vector.
 *
 * @details Grows the knot vector by one and produces a sparse control-point
 *          transform whose non-zero band reflects the local support of the
 *          (p+1) basis functions affected by the new knot. The refined basis
 *          spans the same space as the original, so the curve (or surface)
 *          is preserved when applied to control points. Repeated application
 *          composes via matrix product: callers wanting multiplicity k apply
 *          this function k times.
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
 * @details Implemented as the composition M = R * B * E, where E performs
 *          Bezier extraction by inserting every interior knot to multiplicity
 *          p, B applies per-segment Bernstein degree elevation, and R removes
 *          the previously inserted knots back down to (s_k + 1) where s_k was
 *          their original multiplicity. Continuity at every existing internal
 *          knot is preserved, which is the defining property of p-refinement.
 *          Repeated application composes via matrix product on the original
 *          (homogeneous) control points.
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

} // namespace pyck::bspline_kernel

#endif // PYCK_REFINEMENT_HPP
