#ifndef PYCK_VTK_MULTIBLOCK_HPP
#define PYCK_VTK_MULTIBLOCK_HPP

#include <concepts>
#include <map>
#include <string>
#include <vector>

#include "vtk_export.hpp"
#include "../geometry/patch.hpp"
#include "../types.hpp"

namespace pyck::io
{

/**
 * @brief Write Bezier blocks as a VTK multiblock dataset (`.vtm`).
 *
 * Each block becomes a named, separately selectable `.vtu` child, written into a
 * sibling directory named after the `.vtm` stem; the `.vtm` references them by
 * relative path. Unlike @ref write_bezier_vtu (which merges everything into one
 * piece), the patches stay individually addressable in ParaView.
 *
 * @param path        Output `.vtm` filename.
 * @param blocks      Bezier meshes, one per child dataset.
 * @param block_names Per-block display names (the DataSet `name` attribute);
 *                    empty entries fall back to `block<i>`.
 * @param title       XML header comment.
 * @param binary      Whether each child `.vtu` is raw appended binary.
 */
template <std::floating_point T>
void write_bezier_vtm(const std::string& path,
                      const std::vector<VtkBezierMesh<T>>& blocks,
                      const std::vector<std::string>& block_names,
                      const std::string& title,
                      bool binary);

/**
 * @brief End-to-end multiblock export: extract every patch and write the `.vtm`.
 *
 * @param patches               Patches to export, one child block each.
 * @param point_data            Per-patch name → (num_control_pts × k) field maps.
 * @param point_data_extracted  Per-patch name → (n_ext_cps × k) field maps.
 * @param patch_names           Per-patch display names.
 */
template <std::floating_point T>
void export_bezier_vtm(
    const std::string& path,
    const std::vector<Ptr<Patch<T, 2>>>& patches,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data_extracted,
    const std::vector<std::string>& patch_names,
    const std::string& title,
    bool binary);

} // namespace pyck::io

#endif // PYCK_VTK_MULTIBLOCK_HPP
