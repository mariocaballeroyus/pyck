#ifndef PYCK_PATCH_BOUNDARY_HPP
#define PYCK_PATCH_BOUNDARY_HPP

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
 * @tparam d  Parametric dimension of the parent patch.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class PatchBoundary : public Patch<T, d - 1>
{
public:

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

    /// @brief All parent DOFs on the boundary
    const std::vector<Index>& parent_dofs() const
    { return parent_dofs_; }

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

    /// @brief First non-degenerate span in the parent along the fixed direction
    Index span_fixed() const 
    { return span_fixed_; }

    /// @brief Parameter value at which to evaluate the parent along the fixed direction
    T u_eval_fixed() const 
    { return u_eval_fixed_; }

    /// @brief Sign of the outward normal (+1 or -1) consistent with parametric orientation
    T sign_n() const 
    { return sign_n_; }

    /**
     * @brief Compute outward unit normal at quadrature points.
     *
     * 1D boundary on a 2D surface: normal = boundary_tangent × surface_normal,
     * where surface_normal = parent_tangents[0] × parent_tangents[1].
     */
    ColMatrix<T, 3> eval_normal(
        const std::array<ColMatrix<T, 3>, d - 1>& tangents,
        const std::array<ColMatrix<T, 3>, d>& parent_tangents) const requires(d == 2);

    /**
     * @brief Compute outward unit normal at quadrature points.
     *
     * 2D boundary on a 3D volume: normal = tangent0 × tangent1; parent
     * tangents are unused (the boundary surface defines its own normal).
     */
    ColMatrix<T, 3> eval_normal(
        const std::array<ColMatrix<T, 3>, d - 1>& tangents,
        const std::array<ColMatrix<T, 3>, d>& parent_tangents) const requires(d == 3);

    /**
     * @brief Convert a boundary span index to the flat span index of the parent patch.
     */
    Index parent_flat_span(Index boundary_span) const;

    /**
     * @brief Lift Q boundary parameter values into Q parent parametric coordinates
     *        (the fixed direction is set to u_eval_fixed, the free direction
     *        is filled with the given boundary parameters).
     */
    ColMatrix<T, d> lift_to_parent(const Vector<T>& boundary_pts) const requires(d == 2);

    /**
     * @brief Compute the Q×3 outward unit normal at the boundary quadrature points,
     *        end-to-end from shape function derivatives. Encapsulates the tangent
     *        and surface-normal computations on both the boundary and its parent.
     */
    ColMatrix<T, 3> eval_outward_normal(
        Index boundary_span,
        const std::vector<Matrix<T>>& boundary_derivs,
        Index parent_flat_span,
        const std::vector<Matrix<T>>& parent_derivs) const;

private:

    /// @brief Non-owning const reference to the parent patch, for DOF mapping and geometry access
    Ptr<const Patch<T, d>> parent_;

    /// @brief Parametric direction normal to the boundary
    std::size_t param_dim_;

    /// @brief Whether this is the start (u=0) or end (u=1) boundary
    bool at_start_;

    /// @brief Global DOF indices of the parent patch on the boundary
    std::vector<Index> parent_dofs_;

    /// @brief Fixed span in the parent along the fixed direction
    Index span_fixed_;
    
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
