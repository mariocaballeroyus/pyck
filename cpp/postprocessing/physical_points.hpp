#ifndef PYCK_PHYSICAL_POINTS_HPP
#define PYCK_PHYSICAL_POINTS_HPP

#include "patch.hpp"
#include "tensor_product.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Physical coordinates at every active quadrature point of a patch.
 *
 * @param patch        Patch to sample (1D curve or 2D surface).
 * @param quadrature   Reference quadrature rule (mapped per element).
 * @return ColMatrix<T, 3> of shape (Q_total, 3): packed physical coordinates.
 */
template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> eval_physical_points(
    const Patch<T, d>& patch,
    const QuadratureRule<T, d>& quadrature);

/**
 * @brief Parametric coordinates at every active quadrature point of a patch,
 *        in the same flat-element order as eval_physical_points.
 *
 * @param patch        Patch to sample.
 * @param quadrature   Reference quadrature rule (mapped per element).
 * @return ColMatrix<T, d> of shape (Q_total, d): packed parametric coordinates.
 */
template <std::floating_point T, std::size_t d>
ColMatrix<T, d> eval_parametric_points(
    const Patch<T, d>& patch,
    const QuadratureRule<T, d>& quadrature);

/**
 * @brief Integration measure dV = w * |J| at every active quadrature point,
 *        in the same order as eval_physical_points.
 *
 * @param patch        Patch to sample.
 * @param quadrature   Reference quadrature rule (mapped per element).
 * @return Vector<T> of length Q_total.
 */
template <std::floating_point T, std::size_t d>
Vector<T> eval_integration_measures(
    const Patch<T, d>& patch,
    const QuadratureRule<T, d>& quadrature);

} // namespace pyck

#endif // PYCK_PHYSICAL_POINTS_HPP
