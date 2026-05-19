#ifndef PYCK_PATCH_BOUNDARY_HPP
#define PYCK_PATCH_BOUNDARY_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <concepts>

#include "../types.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

/**
 * @brief Represents a boundary face of a d-dimensional patch.
 *
 * @tparam T  Scalar floating-point type.
 * @tparam d  Parametric dimension of the parent patch.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class PatchBoundary : public Patch<T, d - 1>
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Construct a PatchBoundary
     *
     * @param parent     The parent patch
     * @param param_dim  Parametric direction normal to the boundary (0, 1, …, d−1)
     * @param at_start   True for the boundary at the start of the parametric
     *                   direction (e.g. u = 0), false for the end (e.g. u = 1)
     */
    PatchBoundary(const Ptr<Patch<T, d>>& parent,
                  std::size_t param_dim,
                  bool at_start);

    // === Getters ====================================================================

    /// @brief Returns parent DOFs for assembly, overriding the base class local indices.
    std::vector<Index> assembly_dofs() const override
    { return parent_dofs_; }

    /// @brief Parametric direction normal to this boundary
    std::size_t param_dim() const
    { return param_dim_; }

    /// @brief Whether this is the start or end boundary
    bool at_start() const
    { return at_start_; }

    /// @brief Reference to the parent patch
    const Ptr<const Patch<T, d>>& parent() const
    { return parent_; }

    /// @brief Parent DOFs on the outermost boundary layer (displacement layer).
    std::vector<Index> displacement_dofs() const
    { return parent_->dof_mapper().get_layer_dofs(param_dim_, at_start_, 0); }

    /// @brief Parent DOFs on the adjacent boundary layer (rotation layer).
    std::vector<Index> rotation_dofs() const
    { return parent_->dof_mapper().get_layer_dofs(param_dim_, at_start_, 1); }

    // === Utility Methods ============================================================

    /// @brief Convert a boundary span index to the flat span index of the parent patch.
    Index parent_flat_span(Index boundary_span) const;

    /**
     * @brief Lift Q boundary parameter values into Q parent parametric coordinates.
     *
     * @param boundary_pts Boundary parameter values, shape (Q, d-1). For d=2
     *                     this is a column vector; for d=3 it has two columns
     *                     in the order of the two free parametric directions
     *                     of the parent.
     * @return Matrix of parametric coordinates in the parent patch, shape (Q, d).
     */
    ColMatrix<T, d> lift_to_parent(const ColMatrix<T, d - 1>& boundary_pts) const;

    /**
     * @brief Outward in-surface unit normal at the boundary quadrature points
     *        from precomputed local frames.
     *
     * For d=2 the boundary is a 1D curve in the parent surface; the outward
     * normal lies in the surface and is computed from the boundary tangent
     * crossed with the parent surface normal a_3.
     *
     * For d=3 the boundary is itself a 2D surface in 3D physical space; the
     * outward normal is the boundary's own a_3 = a_1^bd × a_2^bd / ‖…‖,
     * signed by `sign_n_`. The parent local frame is unused.
     */
    ColMatrix<T, 3> eval_outward_normal(const IntrinsicGeometry<T, d - 1>& boundary_local,
                                        const IntrinsicGeometry<T, d>& parent_local) const requires(d == 2);
    ColMatrix<T, 3> eval_outward_normal(const IntrinsicGeometry<T, d - 1>& boundary_local) const requires(d == 3);

private:

    /// @brief Non-owning const reference to the parent patch, for DOF mapping and geometry access
    Ptr<const Patch<T, d>> parent_;

    /// @brief Parametric direction normal to the boundary
    std::size_t param_dim_;

    /// @brief Whether this is the start (u=0) or end (u=1) boundary
    bool at_start_;

    /// @brief Global DOF indices of the parent patch on the boundary
    std::vector<Index> parent_dofs_;

    /// @brief Parameter value at which to evaluate the parent along the fixed direction
    T u_eval_fixed_;

    /// @brief Sign of the outward normal (+1 or -1) consistent with parametric orientation
    T sign_n_;

    /// @brief Parent's span index in the fixed direction (used by parent_flat_span()).
    Index span_fixed_;
};

/**
 * @brief Factory function to create a PatchBoundary.
 * 
 * @tparam T Scalars type
 * @tparam d Parametric dimension of the parent patch
 * @param parent Pointer to the parent patch
 * @param param_dim Parametric dimension normal to the boundary
 * @param at_start True for start boundary, false for end
 * @return A shared pointer to the new PatchBoundary
 */
template <std::floating_point T, std::size_t d>
Ptr<PatchBoundary<T, d>> create_patch_boundary(const Ptr<Patch<T, d>>& parent,
                                         std::size_t param_dim,
                                         bool at_start)
{
    return std::make_shared<PatchBoundary<T, d>>(parent, param_dim, at_start);
}

} // namespace pyck

#endif // PYCK_PATCH_BOUNDARY_HPP
