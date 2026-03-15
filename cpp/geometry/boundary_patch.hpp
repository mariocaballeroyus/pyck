#ifndef PYCK_BOUNDARY_PATCH_HPP
#define PYCK_BOUNDARY_PATCH_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <concepts>

#include "../types.hpp"

namespace pyck
{

// Forward declaration
template <std::floating_point T, std::size_t d> class Patch;

/**
 * @brief Represents a boundary face of a d-dimensional patch.
 *
 * @tparam T  Scalar floating-point type.
 * @tparam d  Parametric dimension of the **parent** patch.
 */
template <std::floating_point T, std::size_t d>
class BoundaryPatch
{
public:

    /**
     * @brief Construct a BoundaryPatch
     *
     * @param parent     The parent patch
     * @param param_dim  Parametric direction normal to the boundary (0, 1, …, d−1)
     * @param at_start   True for the boundary at the start of the parametric
     *                   direction (e.g. u = 0), false for the end (e.g. u = 1)
     */
    BoundaryPatch(const Ptr<Patch<T, d>>& parent,
                  std::size_t param_dim,
                  bool at_start);

    /// @brief All parent DOFs on the two boundary layers (displacement + rotation)
    const std::vector<Index>& boundary_dofs() const { return boundary_dofs_; }

    /// @brief Parent DOFs on the outermost layer (displacement DOFs)
    const std::vector<Index>& displacement_dofs() const { return displacement_dofs_; }

    /// @brief Parent DOFs on the adjacent layer (rotation / slope DOFs)
    const std::vector<Index>& rotation_dofs() const { return rotation_dofs_; }

    /// @brief Parametric direction normal to this boundary
    std::size_t param_dim() const { return param_dim_; }

    /// @brief Whether this is the start or end boundary
    bool at_start() const { return at_start_; }

    /// @brief Reference to the parent patch
    const Ptr<Patch<T, d>>& parent() const { return parent_; }

private:

    /// @brief Pointer to the parent patch (non-owning, for DOF mapping and geometry access)
    Ptr<Patch<T, d>> parent_;

    /// @brief Parametric direction normal to the boundary
    std::size_t param_dim_;

    /// @brief Whether this is the start (u=0) or end (u=1) boundary
    bool at_start_;

    /// @brief Global DOF indices on the boundary layers (both displacement and rotation)
    std::vector<Index> boundary_dofs_;

    /// @brief Global DOF indices on the outermost boundary layer (displacement DOFs)
    std::vector<Index> displacement_dofs_;

    /// @brief Global DOF indices on the adjacent layer (rotation/slope DOFs)
    std::vector<Index> rotation_dofs_;
};

/**
 * @brief Factory function to create a BoundaryPatch.
 * 
 * @tparam T Scalars type
 * @tparam d Parametric dimension of the parent patch
 * @param parent Pointer to the parent patch
 * @param param_dim Parametric dimension normal to the boundary
 * @param at_start True for start boundary, false for end
 * @return A shared pointer to the new BoundaryPatch
 */
template <std::floating_point T, std::size_t d>
Ptr<BoundaryPatch<T, d>> create_boundary(const Ptr<Patch<T, d>>& parent,
                                         std::size_t param_dim,
                                         bool at_start)
{
    return std::make_shared<BoundaryPatch<T, d>>(parent, param_dim, at_start);
}

} // namespace pyck

#endif // PYCK_BOUNDARY_PATCH_HPP
