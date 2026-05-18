#ifndef PYCK_LAPLACE_BELTRAMI_HPP
#define PYCK_LAPLACE_BELTRAMI_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "../types.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

/**
 * @brief Per-quadrature-point auxiliary data for ∂_α(Δ_g f).
 *
 * Storage is nested `std::array` matching the convention used by
 * `IntrinsicGeometry`: symmetric tensor indices populate only their
 * upper-triangular slots. Each `Vector<T>` holds per-quadrature values.
 */
template <std::floating_point T, std::size_t d>
struct LaplaceGradAux
{
    /// (g^{ij})_{,α}, symmetric in (i, j); only i ≤ j filled. G_inv_d[i][j][α].
    std::array<std::array<std::array<Vector<T>, d>, d>, d> G_inv_d;

    /// c^δ = g^{ij} Γ^δ_{ij}.                                   c[δ].
    std::array<Vector<T>, d> c;

    /// (c^δ)_{,α}.                                              c_d[δ][α].
    std::array<std::array<Vector<T>, d>, d> c_d;
};

/**
 * @brief Build a LaplaceGradAux<T, d> at all quadrature points from a
 *        precomputed IntrinsicGeometry. `ig` must have Order 2 populated
 *        (third derivatives required: the basis must have been evaluated at
 *        order ≥ 3). Christoffels are read from `ig.chr` — caller chooses
 *        the level when building the intrinsic geometry.
 */
template <std::floating_point T, std::size_t d>
LaplaceGradAux<T, d>
compute_laplace_grad_aux(const IntrinsicGeometry<T, d>& ig);

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_HPP
