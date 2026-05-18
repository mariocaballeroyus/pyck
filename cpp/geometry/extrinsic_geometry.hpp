#ifndef PYCK_EXTRINSIC_GEOMETRY_HPP
#define PYCK_EXTRINSIC_GEOMETRY_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "../types.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

/**
 * @brief Second fundamental form and shape operator of a surface chart.
 *        Empty until ExtrinsicGeometry::compute_curvature is called.
 */
template <std::floating_point T>
struct SurfaceCurvature
{
    /// @brief b_{αβ} = a_{αβ} · a_3
    std::array<std::array<Vector<T>, 2>, 2> b;

    /// @brief b^α_β = g^{αγ} b_{γβ}
    std::array<std::array<Vector<T>, 2>, 2> b_mixed;
};

/**
 * @brief Extrinsic geometric data of a codim-1 surface in the 3d euclidean space.
 *
 * @note As for now, only 2d parametric geometries are supported, i.e. surface 
 *       patches embedded in the 3d space.
 *
 * @tparam d Parametric dimension.
 * @tparam k Max derivative order.
 */
template <std::floating_point T, std::size_t d, std::size_t k = 2>
    requires (d == 2)
struct ExtrinsicGeometry
{
    // === Normal Vector ==============================================================

    /// Unit normal a_3 = (a_1 × a_2) / √det g.
    ColMatrix<T, 3> n;

    /// Unit normal derivative ∂_β a_3.
    std::array<ColMatrix<T, 3>, 2> n_d1;

    // === Curvature (on-demand) ======================================================

    /// Empty curvature (until `compute_curvature` is called).
    SurfaceCurvature<T> curv;

    // === Constructor ================================================================

    /**
     * @brief Populates the normal vector.
     *
     * @param ig The precomputed intrinsic geometry.
     */
    explicit ExtrinsicGeometry(const IntrinsicGeometry<T, d, k>& ig);

    // === Methods ====================================================================

    /**
     * @brief Populate the curvature.
     *
     * @param ig The precomputed intrinsic geometry.
     */
    void compute_curvature(const IntrinsicGeometry<T, d, k>& ig);
};

} // namespace pyck

#endif // PYCK_EXTRINSIC_GEOMETRY_HPP
