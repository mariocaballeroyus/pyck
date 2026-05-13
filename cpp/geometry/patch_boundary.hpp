#ifndef PYCK_PATCH_BOUNDARY_HPP
#define PYCK_PATCH_BOUNDARY_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <concepts>

#include "../types.hpp"
#include "patch.hpp"
#include "basis_derivs.hpp"
#include "local_frame.hpp"

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

    // === Utility Methods ============================================================

    /// @brief Convert a boundary span index to the flat span index of the parent patch.
    Index parent_flat_span(Index boundary_span) const
    { return parent_span_offset_ + boundary_span * parent_span_stride_; }

    /**
     * @brief Lift Q boundary parameter values into Q parent parametric coordinates.
     *
     * @param boundary_pts Vector of parameter values in the boundary patch
     * @return Matrix of parametric coordinates in the parent patch
     */
    ColMatrix<T, d> lift_to_parent(const Vector<T>& boundary_pts) const requires(d == 2);

    /**
     * @brief Outward in-surface unit normal at the boundary quadrature points
     *        from precomputed local frames.
     */
    ColMatrix<T, 3> eval_outward_normal(const LocalFrame<T, d - 1>& boundary_local,
                                        const LocalFrame<T, d>& parent_local) const requires(d == 2);

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

    /// @brief Offset for flat span indexing
    Index parent_span_offset_;
    
    /// @brief Stride for flat span indexing
    Index parent_span_stride_;
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
