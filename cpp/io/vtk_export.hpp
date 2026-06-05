#ifndef PYCK_VTK_EXPORT_HPP
#define PYCK_VTK_EXPORT_HPP

#include <concepts>
#include <map>
#include <string>
#include <vector>

#include "../geometry/patch.hpp"
#include "../types.hpp"

namespace pyck::io
{

/**
 * @brief One rational-Bezier block: a knot-span-wise VTK_BEZIER_QUADRILATERAL
 *        decomposition of a single surface patch, with its point data.
 *
 * @details Points are laid out cell-by-cell in VTK higher-order quad ordering
 *          (4 corners, then per-edge interiors, then the face block), so the
 *          connectivity is simply `0, 1, …, n_points-1`. Every cell carries
 *          `(p+1)·(q+1)` points; `offsets[c]` is the one-past-the-end index of
 *          cell `c`'s points.
 */
template <std::floating_point T>
struct VtkBezierMesh
{
    ColMatrix<T, 3>           points;            ///< (n_points × 3) physical coords.
    Vector<T>                 rational_weights;  ///< (n_points) NURBS denominators.
    std::vector<std::string>  field_names;       ///< Point-data array names.
    std::vector<Matrix<T>>    field_values;      ///< Parallel: each (n_points × ncomp).
    std::vector<Index>        offsets;           ///< (n_cells) cumulative point counts.
    IndexMatrix               degrees;           ///< (n_cells × 3) HigherOrderDegrees.

    Index num_points() const { return static_cast<Index>(points.rows()); }
    Index num_cells()  const { return static_cast<Index>(offsets.size()); }
};

/**
 * @brief Bezier-extract one surface patch and pack it into a @ref VtkBezierMesh.
 *
 * The geometry and every `point_data` field are pushed onto the per-element
 * Bernstein basis with the same (rational, for NURBS) extraction operator, so
 * the surface and fields are represented exactly. `point_data_extracted` fields
 * are taken as already living on the extracted basis (the Greville
 * quasi-interpolant recipe for derived fields such as strain/stress).
 *
 * @param patch                 Surface patch to extract.
 * @param point_data            name → (num_control_pts × k) arrays on the patch basis.
 * @param point_data_extracted  name → (n_ext_cps × k) arrays on the extracted basis.
 * @return The packed Bezier mesh for this patch.
 */
template <std::floating_point T>
VtkBezierMesh<T> extract_bezier_mesh(
    const Patch<T, 2>& patch,
    const std::map<std::string, Matrix<T>>& point_data,
    const std::map<std::string, Matrix<T>>& point_data_extracted);

/**
 * @brief Greville parametric coordinates of the Bezier-extracted basis.
 *
 * @return `(n_ext_cps × 2)` array of `(u, v)` anchor points in u-fastest order,
 *         matching the extracted CP layout. Sampling a derived field here and
 *         passing the values as `point_data_extracted` is the standard
 *         quasi-interpolation recipe.
 */
template <std::floating_point T>
ColMatrix<T, 2> bezier_anchor_params(const Patch<T, 2>& patch);

/**
 * @brief Write one or more Bezier blocks to a single `.vtu` file.
 *
 * Blocks are concatenated as independent cells in one `UnstructuredGrid` piece.
 *
 * @param path   Output filename.
 * @param blocks Bezier meshes to write. All must share the same field name set
 *               and per-field component counts.
 * @param title  XML header comment.
 * @param binary If true, arrays are written as raw appended binary (compact,
 *               fast to load); otherwise inline ASCII.
 */
template <std::floating_point T>
void write_bezier_vtu(const std::string& path,
                      const std::vector<VtkBezierMesh<T>>& blocks,
                      const std::string& title,
                      bool binary);

/**
 * @brief End-to-end export: extract every patch and write the combined `.vtu`.
 *
 * @param patches               Patches to export (concatenated into one file).
 * @param point_data            Per-patch name → (num_control_pts × k) field maps.
 * @param point_data_extracted  Per-patch name → (n_ext_cps × k) field maps.
 */
template <std::floating_point T>
void export_bezier_vtu(
    const std::string& path,
    const std::vector<Ptr<Patch<T, 2>>>& patches,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data_extracted,
    const std::string& title,
    bool binary);

} // namespace pyck::io

#endif // PYCK_VTK_EXPORT_HPP
