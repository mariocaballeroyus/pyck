#ifndef PYCK_PHYSICAL_POINTS_HPP
#define PYCK_PHYSICAL_POINTS_HPP

#include "patch.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Physical coordinates at every active quadrature point of a patch, in
 *        the same flat-element order the assembly will visit them.
 *
 * Maps the reference quadrature rule into each element's parametric span,
 * evaluates basis values at the mapped points (order 0), and matrix-multiplies
 * by the active control points. Zero-volume spans (degenerate clamped knot
 * intervals) are skipped. Used to pre-evaluate user-supplied load functions
 * `f(x, y, z)` at the right qpts before assembly.
 *
 * @param patch        Patch to sample (1D curve or 2D surface).
 * @param quadrature   Reference quadrature rule (mapped per element).
 * @return ColMatrix<T, 3> of shape (Q_total, 3): packed physical coordinates.
 */
template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> eval_physical_points(
    const Patch<T, d>& patch,
    const QuadratureRule<T, d>& quadrature);

} // namespace pyck

#endif // PYCK_PHYSICAL_POINTS_HPP
