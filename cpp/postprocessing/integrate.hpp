#ifndef PYCK_INTEGRATE_HPP
#define PYCK_INTEGRATE_HPP

#include <functional>
#include <vector>
#include <Eigen/Dense>

#include "patch.hpp"
#include "basis_derivs.hpp"
#include "local_frame.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Integrate a user-defined per-quadrature-point functional over a patch.
 *
 * Loops over elements; per element, evaluates the active (p+1)^d basis
 * derivatives via `eval_basis` plus the local frame `LocalFrame<T,d>`
 * (tangents, metric, inverse metric, area element); also returns the physical
 * coordinates of the mapped quadrature points and (optionally) gathers the
 * active DOF block from `cp_values`. The user-provided `integrand` returns a
 * (Q, n_components) matrix of per-quadrature-point values; the integrator
 * multiplies by dV = w * |J| and accumulates each component independently.
 *
 * Memory is bounded: only (p+1)^d basis functions are materialised per
 * element — no dense (Q x n_basis_total) matrix is ever allocated.
 *
 * @param patch        Patch to integrate over.
 * @param quadrature   Reference quadrature rule (mapped per element).
 * @param order        Basis-derivative order to request from `eval_basis`.
 * @param cp_values    Per-control-point DOFs in node-major order, length
 *                     n_cp * ndof. Pass an empty vector when the integrand
 *                     does not need DOF data.
 * @param ndof         Number of DOFs per control point in `cp_values`.
 *                     Ignored when `cp_values` is empty.
 * @param integrand    Callback returning per-Q values. Signature:
 *                       integrand(phys_pts, basis, local, u_local) -> (Q, n)
 *                     where `u_local` has length K_active * ndof, or is empty
 *                     when `cp_values` is empty.
 * @return Vector<T> of length n_components — the per-component integrals.
 */
template <std::floating_point T, std::size_t d>
Vector<T> integrate_on_patch(
    const Patch<T, d>& patch,
    const QuadratureRule<T, d>& quadrature,
    std::size_t order,
    const Vector<T>& cp_values,
    std::size_t ndof,
    const std::function<Matrix<T>(
        const ColMatrix<T, 3>&,
        const BasisDerivs<T, d>&,
        const LocalFrame<T, d>&,
        const Vector<T>&)>& integrand);

} // namespace pyck

#endif // PYCK_INTEGRATE_HPP
