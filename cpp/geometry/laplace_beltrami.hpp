#ifndef PYCK_LAPLACE_BELTRAMI_HPP
#define PYCK_LAPLACE_BELTRAMI_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../types.hpp"

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
 * @brief Build a LaplaceGradAux<T, d> at all quadrature points from raw
 *        base-vector data: the position derivatives (must be populated to
 *        order 3 — the basis evaluated at order ≥ 3) and the contravariant
 *        metric. The connection is formed internally from base-vector dot
 *        products `A_ε·A_{,ij}` raised by `g_inv` — no Christoffel array is
 *        needed. Decoupled from `ElementValues` so this stays a leaf header
 *        (no circular include) consumable by the kernel layer.
 */
template <std::floating_point T, std::size_t d>
LaplaceGradAux<T, d>
compute_laplace_grad_aux(const std::vector<ColMatrix<T, 3>>& position_data,
                         const Matrix<T>& g_inv_data);

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_HPP
