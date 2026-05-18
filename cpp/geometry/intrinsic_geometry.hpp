#ifndef PYCK_INTRINSIC_GEOMETRY_HPP
#define PYCK_INTRINSIC_GEOMETRY_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "../basis/basis_values.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Christoffel symbols of the second kind and their first derivatives,
 *        stored in packed column-major matrices indexed by (k, i, j[, γ]).
 *
 * Storage layout:
 *   - @ref Gamma_data: shape Q × (d · n_d2), col-major. Column index packed as
 *     `k · n_d2 + pack2<d>(i, j)`, where n_d2 = d(d+1)/2.
 *   - @ref Gamma_d1_data: shape Q × (d · n_d2 · d). Column index packed as
 *     `k · (n_d2 · d) + pack2<d>(i, j) · d + γ`.
 *
 * Element / interior-geometry code accesses via the named methods, which
 * canonicalise the (i, j) pair to upper-triangular form. Empty if
 * @ref IntrinsicGeometry::compute_christoffels was not called.
 */
template <std::floating_point T, std::size_t d>
struct Christoffels
{
    /// Packed Γ^k_{ij}(q) data. Empty until compute_christoffels.
    Matrix<T> Gamma_data;

    /// Packed ∂_γ Γ^k_{ij}(q) data. Empty unless basis order ≥ 3.
    Matrix<T> Gamma_d1_data;

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
};

/**
 * @brief Bonnet-intrinsic geometric data of a d-dimensional parametric chart,
 *        stored in Gismo-style packed buffers (one per derivative order for
 *        position derivatives; column-packed for metric / Christoffels).
 *
 * Position derivatives storage @ref position_data is a `std::vector` of
 *   `(Q · n_k) × 3` col-major @c ColMatrix<T,3>, one entry per order
 *   `0 ≤ k ≤ basis.order()`. Within each order, the multi-index slice for
 *   packed-index `m` lives at rows `m·Q .. (m+1)·Q - 1`.
 *
 * Metric and Jacobian are always populated. `position_data` has size
 *   `basis.order() + 1`; accessing higher-order derivatives is undefined
 *   behaviour. Christoffels are populated by an explicit call.
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

    /// Christoffel symbols (populated by compute_christoffels()).
    Christoffels<T, d> chr;

    // === Constructors ===============================================================

    IntrinsicGeometry() = default;

    /**
     * @brief Populate position, metric, inverse metric, Jacobian and position
     *        derivatives up to @c basis.order().
     */
    IntrinsicGeometry(const BasisValues<T, d>& basis,
                      const ColMatrix<T, 3>& act_pts);

    // === Methods ====================================================================

    /**
     * @brief Populate @ref chr from the position derivatives and metric data.
     *        Idempotent (re-calls overwrite).
     *
     * Requires position derivatives of at least order 2 for @c chr.Gamma; if
     * the underlying basis order is at least 3, @c chr.Gamma_d1 is also
     * populated.
     */
    void compute_christoffels();

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
