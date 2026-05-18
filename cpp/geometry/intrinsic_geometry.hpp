#ifndef PYCK_INTRINSIC_GEOMETRY_HPP
#define PYCK_INTRINSIC_GEOMETRY_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "../types.hpp"
#include "basis_derivs.hpp"

namespace pyck
{

/**
 * @brief Christoffel symbols of the second kind and derivatives up to `order`
 *        on a d-dimensional chart.
 */
template <std::floating_point T, std::size_t d>
struct Christoffels
{
    /// Γ^k_{αβ}, sym in (α,β); [k][α][β] with α ≤ β filled.
    std::array<std::array<std::array<Vector<T>, d>, d>, d> Gamma;

    /// ∂_γ Γ^k_{αβ}, sym in (α,β); [k][α][β][γ] with α ≤ β filled, γ free.
    std::array<std::array<std::array<std::array<Vector<T>, d>, d>, d>, d> Gamma_d1;
};

/**
 * @brief Bonnet-intrinsic geometric data of a d-dimensional parametric chart.
 *
 * @tparam T Floating point type.
 * @tparam d Parametric dimension.
 * @tparam k Highest derivative order to compute.
 */
template <std::floating_point T, std::size_t d, std::size_t k = 3>
struct IntrinsicGeometry
{

    // === Position and Covariant Basis ===============================================

    /// @brief Position x(ξ)
    ColMatrix<T, 3> x;

    /// @brief Covariant basis a_α = ∂_α x
    std::array<ColMatrix<T, 3>, d> a;

    /// @brief Symmetric second derivatives a_{αβ}; α ≤ β filled
    std::array<std::array<ColMatrix<T, 3>, d>, d> a_d1;

    /// @brief Symmetric third derivatives a_{αβγ}; α ≤ β ≤ γ filled
    std::array<std::array<std::array<ColMatrix<T, 3>, d>, d>, d> a_d2;

    // === Metrics and Measures =======================================================

    /// @brief Covariant metric g_{αβ} = a_α · a_β; sym, α ≤ β filled
    std::array<std::array<Vector<T>, d>, d> g;

    /// @brief Contravariant metric g^{αβ}
    std::array<std::array<Vector<T>, d>, d> g_inv;

    /// @brief Jacobian √det g
    Vector<T> jac;

    // === Christoffels (on-demand) ===================================================

    /// Empty until @ref compute_christoffels is called.
    Christoffels<T, d> chr;

    // === Constructors ===============================================================

    /**
     * @brief Populates position, metric, inverse metric, Jacobian and
              covariant basis and derivatives up to `k`.
     *
     *        `k >= 1` for `a`
     *        `k >= 2` for `a_d1`
     *        `k >= 3` for `a_d2`
     */
    IntrinsicGeometry(const BasisDerivs<T, d>& basis,
                      const ColMatrix<T, 3>& act_pts);

    // === Methods ====================================================================

    /**
     * @brief Populate `chr` from the basis and metric data. Idempotent
     *        (re-calls overwrite).
     *
     * Computes Christoffel symbols and derivatives up to the order specified by `k`:
     *        `k >= 1` for `chr.Gamma`
     *        `k >= 2` for `chr.Gamma_d1`
     */
    void compute_christoffels();
};

// === Helpers ========================================================================

/**
 * @brief Pack the upper-triangular metric components into a column vector at
 *        q-point `q`. For d=2 this is (g11, g12, g22)ᵀ; for d=3
 *        (g11, g12, g13, g22, g23, g33)ᵀ.
 */
template <std::floating_point T, std::size_t d>
Eigen::Matrix<T, d * (d + 1) / 2, 1>
g_voigt(const IntrinsicGeometry<T, d>& ig, Index q)
{
    Eigen::Matrix<T, d * (d + 1) / 2, 1> v;
    std::size_t flat = 0;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = i; j < d; ++j)
            v(flat++) = ig.g[i][j](q);
    return v;
}

/// Same packing as @ref g_voigt, applied to the inverse metric.
template <std::floating_point T, std::size_t d>
Eigen::Matrix<T, d * (d + 1) / 2, 1>
g_inv_voigt(const IntrinsicGeometry<T, d>& ig, Index q)
{
    Eigen::Matrix<T, d * (d + 1) / 2, 1> v;
    std::size_t flat = 0;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = i; j < d; ++j)
            v(flat++) = ig.g_inv[i][j](q);
    return v;
}

} // namespace pyck

#endif // PYCK_INTRINSIC_GEOMETRY_HPP
