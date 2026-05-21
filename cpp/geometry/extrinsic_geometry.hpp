#ifndef PYCK_EXTRINSIC_GEOMETRY_HPP
#define PYCK_EXTRINSIC_GEOMETRY_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "../multi_index.hpp"
#include "../types.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

/**
 * @brief Per-quantity update flags for @ref ExtrinsicGeometry. Each flag gates
 *        both the storage allocation and the per-element computation of the
 *        associated quantity, mirroring deal.II's `UpdateFlags` pattern.
 *
 * Flags are interpreted on a best-effort basis: `reinit` silently clamps each
 * flag against the source `IntrinsicGeometry`'s available order
 * (`normal_derivatives` and `curvature` both need ig.order() ≥ 2). Default-
 * constructed flags therefore yield the maximum the source supports.
 *
 * `ExtrinsicGeometry` is only meaningful for `d == 2` (surfaces in 3D);
 * `reinit` is a no-op for other `d`.
 */
struct ExtrinsicGeometryFlags
{
    /// Compute normal derivatives ∂_β a_3.
    bool normal_derivatives = true;

    /// Compute second fundamental form b_{αβ} and shape operator b^α_β.
    bool curvature          = true;
};

/**
 * @brief Extrinsic geometric data of a codim-1 surface in 3d Euclidean space.
 *
 *        Normal derivatives storage @ref n_d1_data is a `(Q · 2) × 3`
 *        col-major @c ColMatrix<T,3>: the dir-`β` slice lives at rows
 *        `β·Q .. (β+1)·Q − 1`. Curvature storage:
 *          - @ref b_data:       shape Q × 3 col-major; symmetric in (α, β);
 *            columns packed via `pack2<2>(α, β)`.
 *          - @ref b_mixed_data: shape Q × 4 col-major; not symmetric; column
 *            index `α · 2 + β`.
 *
 * The class is templated over `d` for uniform composition with
 * `ElementValues<T, d>`, but only `d == 2` carries meaningful data. For other
 * `d`, `reinit` is a no-op and all storage stays empty.
 *
 * @tparam T Floating point type.
 * @tparam d Parametric dimension (only d == 2 is computed).
 */
template <std::floating_point T, std::size_t d>
struct ExtrinsicGeometry
{
    /// Unit normal a_3 = (a_1 × a_2) / √det g, shape Q × 3.
    ColMatrix<T, 3> n;

    /// Packed normal derivatives ∂_β a_3: (Q · 2) × 3 col-major. Populated
    /// when `flags.normal_derivatives`.
    ColMatrix<T, 3> n_d1_data;

    /// Packed b_{αβ} = a_{αβ} · a_3, populated when `flags.curvature`.
    /// Symmetric in (α, β).
    Matrix<T> b_data;

    /// Packed b^α_β = g^{αγ} b_{γβ}, populated when `flags.curvature`.
    /// Not symmetric.
    Matrix<T> b_mixed_data;

    // === Constructors ===============================================================

    ExtrinsicGeometry() = default;

    /**
     * @brief Populate the unit normal and the quantities selected by @p flags.
     *        Delegates to @ref reinit.
     */
    explicit ExtrinsicGeometry(const IntrinsicGeometry<T, d>& ig,
                               ExtrinsicGeometryFlags flags = {})
    {
        reinit(ig, flags);
    }

    // === Methods ====================================================================

    /**
     * @brief Refresh the unit normal and the quantities selected by @p flags
     *        on existing storage.
     *
     * No-op for `d != 2`. Internal `resize` calls are no-ops once the buffers
     * have been sized by a prior `reinit` call with matching Q and flags,
     * making subsequent refreshes allocation-free.
     */
    void reinit(const IntrinsicGeometry<T, d>& ig,
                ExtrinsicGeometryFlags flags = {});

    // === Properties =================================================================

    /// Number of quadrature points.
    Index Q() const { return n.rows(); }

    /// @brief Whether normal derivatives are populated.
    bool has_n_d1() const { return n_d1_data.rows() > 0; }

    // === Accessors ==================================================================

    /// ∂_β a_3 — Q × 3 view.
    auto n_d1(Index beta) const
    {
        const Index q = Q();
        return n_d1_data.middleRows(beta * q, q);
    }

    /// b_{αβ}(q). Symmetric in (α, β).
    auto b(Index alpha, Index beta) const
    { return b_data.col(pack2<2>(alpha, beta)); }

    /// b^α_β(q). Not symmetric.
    auto b_mixed(Index alpha, Index beta) const
    { return b_mixed_data.col(alpha * 2 + beta); }
};

} // namespace pyck

#endif // PYCK_EXTRINSIC_GEOMETRY_HPP
