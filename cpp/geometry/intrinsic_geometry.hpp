#ifndef PYCK_INTRINSIC_GEOMETRY_HPP
#define PYCK_INTRINSIC_GEOMETRY_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "../basis/tensor_product.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Per-quantity update flags for @ref IntrinsicGeometry. Each flag gates
 *        both the storage allocation and the per-element computation of the
 *        associated quantity, mirroring deal.II's `UpdateFlags` pattern.
 *
 * Flags are interpreted on a best-effort basis: `reinit` silently clamps each
 * flag against the basis's available derivative orders (`metric` needs order
 * ≥ 1, `christoffels` ≥ 2, `christoffels_d1` ≥ 3). Default-constructed flags
 * therefore yield the maximum the basis supports — preserving the previous
 * "always compute Christoffels" semantics.
 */
struct IntrinsicGeometryFlags
{
    /// Compute covariant metric, contravariant metric and Jacobian.
    bool metric          = true;

    /// Compute Christoffel symbols Γ^k_{ij}.
    bool christoffels    = true;

    /// Compute first derivatives of Christoffel symbols ∂_γ Γ^k_{ij}.
    bool christoffels_d1 = true;
};

/**
 * @brief Bonnet-intrinsic geometric data of a d-dimensional parametric chart,
 *        stored in Gismo-style packed buffers (one per derivative order for
 *        position derivatives; column-packed for metric / Christoffels).
 *
 * Position derivatives storage @ref position_data is a `std::vector` of
 *   `(Q · n_k) × 3` col-major @c ColMatrix<T,3>, one entry per order
 *   `0 ≤ k ≤ (basis.size() - 1)`. Within each order, the multi-index slice
 *   for packed-index `m` lives at rows `m·Q .. (m+1)·Q - 1`.
 *
 * Population of metric, Christoffels and Γ_d1 is controlled by
 * `IntrinsicGeometryFlags` passed to `reinit`.
 *
 * Christoffels storage:
 *   - @ref Gamma_data: shape Q × (d · n_d2), col-major. Column index packed as
 *     `k · n_d2 + pack2<d>(i, j)`, where n_d2 = d(d+1)/2.
 *   - @ref Gamma_d1_data: shape Q × (d · n_d2 · d). Column index packed as
 *     `k · (n_d2 · d) + pack2<d>(i, j) · d + γ`.
 *
 * @tparam T Floating point type.
 * @tparam d Parametric dimension.
 */
template <std::floating_point T, std::size_t d>
struct IntrinsicGeometry
{
    /// Per-order packed position-derivative storage (size = order + 1).
    std::vector<ColMatrix<T, 3>> position_data;

    /// Packed covariant metric: shape Q × n_metric col-major, columns indexed
    /// by `pack2<d>(i, j)`. n_metric = d(d+1)/2.
    Matrix<T> g_data;

    /// Packed contravariant metric: same shape and packing as @ref g_data.
    Matrix<T> g_inv_data;

    /// Jacobian √det g, Q-length.
    Vector<T> jac;

    /// Packed Γ^k_{ij}(q) data, populated when `flags.christoffels`.
    Matrix<T> Gamma_data;

    /// Packed ∂_γ Γ^k_{ij}(q) data, populated when `flags.christoffels_d1`.
    Matrix<T> Gamma_d1_data;

    // === Constructors ===============================================================

    IntrinsicGeometry() = default;

    /**
     * @brief Populate position derivatives and the quantities selected by
     *        @p flags. Delegates to @ref reinit.
     */
    IntrinsicGeometry(const std::vector<Matrix<T>>& basis,
                      const ColMatrix<T, 3>& act_pts,
                      IntrinsicGeometryFlags flags = {})
    {
        reinit(basis, act_pts, flags);
    }

    // === Methods ====================================================================

    /**
     * @brief Refresh position derivatives (up to the basis buffer's maximum
     *        order) and the quantities selected by @p flags on existing
     *        storage.
     *
     * Internal `resize` calls are no-ops once the buffers have been sized by
     * a prior `reinit` call with matching Q, order and flags, making
     * subsequent refreshes allocation-free.
     *
     * @param basis    Per-order packed basis values + derivatives.
     * @param act_pts  Active control points (one row per active basis fn).
     * @param flags    Which derived quantities to compute. The basis must
     *                 carry enough orders to satisfy them (debug-asserted).
     */
    void reinit(const std::vector<Matrix<T>>& basis,
                const ColMatrix<T, 3>& act_pts,
                IntrinsicGeometryFlags flags = {});

    // === Properties =================================================================

    /// Number of quadrature points.
    Index Q() const { return jac.size(); }

    /// Highest derivative order of position data stored.
    Index order() const
    {
        return position_data.empty() ? Index(0)
                                     : Index(position_data.size()) - 1;
    }

    // === Position Accessors (Q × 3 views) ===========================================

    /// Position x(ξ).
    auto x() const                              { return view_pos(0, 0); }

    /// Covariant basis a_α = ∂_α x.
    auto a(Index i) const                       { return view_pos(1, i); }

    /// Symmetric second derivative a_{αβ} = ∂_{αβ} x.
    auto a_d1(Index i, Index j) const           { return view_pos(2, pack2<d>(i, j)); }

    /// Symmetric third derivative a_{αβγ} = ∂_{αβγ} x.
    auto a_d2(Index i, Index j, Index k) const  { return view_pos(3, pack3<d>(i, j, k)); }

    // === Metric Accessors (Q-length views) ==========================================

    /// Covariant metric g_{αβ}(q). Symmetric in (α, β).
    auto g(Index i, Index j) const     { return g_data.col(pack2<d>(i, j)); }

    /// Contravariant metric g^{αβ}(q). Symmetric in (α, β).
    auto g_inv(Index i, Index j) const { return g_inv_data.col(pack2<d>(i, j)); }

    // === Christoffel Accessors (Q-length views) =====================================

    /// Γ^k_{ij}(q) — Q-length view; symmetric in (i, j).
    auto Gamma(Index k, Index i, Index j) const
    {
        constexpr Index n_d2 = d * (d + 1) / 2;
        return Gamma_data.col(k * n_d2 + pack2<d>(i, j));
    }

    /// ∂_γ Γ^k_{ij}(q) — Q-length view; symmetric in (i, j), free in γ.
    auto Gamma_d1(Index k, Index i, Index j, Index gam) const
    {
        constexpr Index n_d2 = d * (d + 1) / 2;
        return Gamma_d1_data.col(k * (n_d2 * d) + pack2<d>(i, j) * d + gam);
    }

private:

    /// Build a Q × 3 view onto packed-position @p packed within order-@p ord storage.
    auto view_pos(Index ord, Index packed) const
    {
        const Index q = Q();
        return position_data[ord].middleRows(packed * q, q);
    }
};

// === Helpers ========================================================================

/**
 * @brief Pack the upper-triangular metric components into a column vector at
 *        q-point @p q. For d=2 this is (g11, g12, g22)ᵀ; for d=3
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
            v(flat++) = ig.g(i, j)(q);
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
            v(flat++) = ig.g_inv(i, j)(q);
    return v;
}

} // namespace pyck

#endif // PYCK_INTRINSIC_GEOMETRY_HPP
