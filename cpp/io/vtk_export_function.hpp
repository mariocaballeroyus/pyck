#ifndef PYCK_VTK_EXPORT_FUNCTION_HPP
#define PYCK_VTK_EXPORT_FUNCTION_HPP

#include <concepts>
#include <map>
#include <string>
#include <vector>

#include "vtk_export.hpp"
#include "../postprocessing/function.hpp"
#include "../types.hpp"

namespace pyck::io
{

/**
 * @brief Sample a @ref Function at a patch's Bezier-extracted Greville anchors.
 *
 * @return `(n_ext_cps × k)` field values in the extracted u-fastest layout,
 *         ready to pass as `point_data_extracted` to @ref export_bezier_vtu.
 */
template <std::floating_point T>
Matrix<T> sample_at_anchors(const Function<T, 2>& field, const Patch<T, 2>& patch)
{
    return field(bezier_anchor_params<T>(patch));
}

/**
 * @brief Function-driven export: sample each @ref Function (displacement,
 *        strain, stress, …) at the extracted anchors and write it as an exact
 *        derived field.
 *
 * This is the headless C++ path — a solver can dump derived fields with no
 * Python round-trip. It composes with `point_data` for fields that already live
 * on the control-point basis.
 *
 * @param functions Per-patch name → Function maps; each is sampled on its patch.
 */
template <std::floating_point T>
void export_bezier_vtu(
    const std::string& path,
    const std::vector<Ptr<Patch<T, 2>>>& patches,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data,
    const std::vector<std::map<std::string, Function<T, 2>>>& functions,
    const std::string& title,
    bool binary)
{
    if (functions.size() != patches.size())
        throw std::runtime_error(
            "vtk_export: per-patch function list must match the patch count.");

    std::vector<std::map<std::string, Matrix<T>>> extracted(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i)
        for (const auto& [name, field] : functions[i])
            extracted[i].emplace(name, sample_at_anchors<T>(field, *patches[i]));

    export_bezier_vtu<T>(path, patches, point_data, extracted, title, binary);
}

} // namespace pyck::io

#endif // PYCK_VTK_EXPORT_FUNCTION_HPP
