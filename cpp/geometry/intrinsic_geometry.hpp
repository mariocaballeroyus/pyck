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
 * @brief Bonnet-intrinsic geometric data of a d-dimensional parametric chart,
 *        stored in Gismo-style packed buffers (one per derivative order for
 *        position derivatives; column-packed for metric / Christoffels).
 *
 * Position derivatives storage @ref position_data is a `std::vector` of
 *   `(Q · n_k) × 3` col-major @c ColMatrix<T,3>, one entry per order
 *   `0 ≤ k ≤ (basis.size() - 1)`. Within each order, the multi-index slice
 *   for packed-index `m` lives at rows `m·Q .. (m+1)·Q - 1`.
 *
 * Which quantities are populated is determined entirely by the basis
 * derivative order supplied to `reinit`:
 *   - order ≥ 1 → covariant + contravariant metric and Jacobian
 *   - order ≥ 2 → Christoffel symbols Γ^k_{ij}
 *   - order ≥ 3 → Christoffel symbol derivatives ∂_γ Γ^k_{ij}
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
     * @brief Populate position derivatives and every derived quantity up to
     *        @p order. Delegates to @ref reinit.
     */
    IntrinsicGeometry(const std::vector<Matrix<T>>& basis,
                      const ColMatrix<T, 3>& act_pts,
                      const Index order)
    {
        reinit(basis, act_pts, order);
    }

    // === Methods ====================================================================

    /**
     * @brief Refresh position derivatives and every derived quantity up to
     *        @p order (see class doc for the order ↔ quantity ladder). The
     *        basis must carry at least @p order derivatives (debug-asserted).
     *
     * Internal `resize` calls are no-ops once the buffers have been sized by
     * a prior `reinit` call with matching Q and order, making subsequent
     * refreshes allocation-free.
     *
     * @param basis    Per-order packed basis values + derivatives.
     * @param act_pts  Active control points (one row per active basis fn).
     * @param order    Maximum derivative order to evaluate (≤ basis.size()-1).
     */
    void reinit(const std::vector<Matrix<T>>& basis,
                const ColMatrix<T, 3>& act_pts,
                const Index order);

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

    /// Pass 1 — fill @ref position_data from basis values and active control
    /// points (covariant tangents and higher-order position derivatives) up
    /// to derivative order @p order. The basis may carry more entries than
    /// @p order + 1; only the first @p order + 1 are read.
    void compute_position_derivatives_(const std::vector<Matrix<T>>& basis,
                                       const ColMatrix<T, 3>& act_pts,
                                       const Index order);

    /// Pass 2 — fill @ref g_data, @ref g_inv_data and @ref jac. Requires
    /// @ref compute_position_derivatives_ to have been called with order ≥ 1.
    void compute_metric_();

    /// Pass 3 — fill @ref Gamma_data. Requires Pass 2 to have been called
    /// and basis order ≥ 2.
    void compute_christoffels_();

    /// Pass 4 — fill @ref Gamma_d1_data. Requires Pass 2 to have been called
    /// and basis order ≥ 3. Recomputes the per-q intermediates `g_inv_full`
    /// and `aup` that Pass 3 also uses; the redundant arithmetic is small
    /// (~d² mults/adds per qpt) and worth the cleaner separation.
    void compute_christoffels_d1_();

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
