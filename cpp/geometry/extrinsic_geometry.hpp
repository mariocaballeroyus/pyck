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
 * @brief Second fundamental form and shape operator of a surface chart,
 *        stored in packed column-major matrices.
 *
 * Storage:
 *   - @ref b_data:       shape Q × 3 col-major; symmetric in (α, β); columns
 *     packed via `pack2<2>(α, β)`.
 *   - @ref b_mixed_data: shape Q × 4 col-major; not symmetric; column index
 *     `α · 2 + β`.
 *
 * Empty until @ref ExtrinsicGeometry::compute_curvature is called.
 */
template <std::floating_point T>
struct SurfaceCurvature
{
    /// Packed b_{αβ} = a_{αβ} · a_3. Symmetric in (α, β).
    Matrix<T> b_data;

    /// Packed b^α_β = g^{αγ} b_{γβ}. Not symmetric.
    Matrix<T> b_mixed_data;

    /// b_{αβ}(q). Symmetric in (α, β).
    auto b(Index alpha, Index beta) const
    { return b_data.col(pack2<2>(alpha, beta)); }

    /// b^α_β(q). Not symmetric.
    auto b_mixed(Index alpha, Index beta) const
    { return b_mixed_data.col(alpha * 2 + beta); }
};

/**
 * @brief Extrinsic geometric data of a codim-1 surface in 3d Euclidean space.
 *
 *        Position derivatives storage @ref n_d1_data is a `(Q · 2) × 3`
 *        col-major @c ColMatrix<T,3>: the dir-`β` slice lives at rows
 *        `β·Q .. (β+1)·Q − 1`.
 *
 * @note Currently only 2-parametric geometries supported (surfaces in 3D).
 *
 * @tparam T Floating point type.
 * @tparam d Parametric dimension. Must be 2.
 */
template <std::floating_point T, std::size_t d>
    requires (d == 2)
struct ExtrinsicGeometry
{
    /// Unit normal a_3 = (a_1 × a_2) / √det g, Q × 3.
    ColMatrix<T, 3> n;

    /// Packed normal derivatives ∂_β a_3: (Q · 2) × 3 col-major. Empty if
    /// the source IntrinsicGeometry has order < 2.
    ColMatrix<T, 3> n_d1_data;

    /// Curvature (empty until @ref compute_curvature is called).
    SurfaceCurvature<T> curv;

    // === Constructor ================================================================

    /**
     * @brief Populate the normal vector (always) and normal derivatives if
     *        the source intrinsic geometry has order ≥ 2.
     */
    explicit ExtrinsicGeometry(const IntrinsicGeometry<T, d>& ig);

    // === Methods ====================================================================

    /**
     * @brief Populate the second fundamental form and shape operator.
     *        Requires @c ig.order() ≥ 2.
     */
    void compute_curvature(const IntrinsicGeometry<T, d>& ig);

    // === Properties =================================================================

    /// Number of quadrature points.
    Index Q() const { return n.rows(); }

    /// @brief Whether normal derivatives are populated (true if ig had order ≥ 2).
    bool has_n_d1() const { return n_d1_data.rows() > 0; }

    // === Accessors ==================================================================

    /// ∂_β a_3 — Q × 3 view.
    auto n_d1(Index beta) const
    {
        const Index q = Q();
        return n_d1_data.middleRows(beta * q, q);
    }
};

} // namespace pyck

#endif // PYCK_EXTRINSIC_GEOMETRY_HPP
