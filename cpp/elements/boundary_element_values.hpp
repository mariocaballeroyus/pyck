#ifndef PYCK_BOUNDARY_ELEMENT_VALUES_HPP
#define PYCK_BOUNDARY_ELEMENT_VALUES_HPP

#include <concepts>
#include <cstddef>

#include "../geometry/patch_boundary.hpp"
#include "../types.hpp"
#include "element_values.hpp"

namespace pyck
{

/**
 * @brief Per-span evaluation workspace for a PatchBoundary.
 *
 * @details Boundary integration is intrinsically a two-basis problem: the
 *          integrand involves both the boundary's own (d-1)-dim basis (for
 *          multipliers, the boundary measure, normal-traction terms) and
 *          the parent's d-dim basis evaluated at the lifted boundary
 *          quadrature points (because primary unknowns live on the parent).
 *          Composes two `ElementValues`: `boundary_vals_` (the (d-1)-dim
 *          workspace, driven by the boundary's own quadrature) and
 *          `parent_vals_` (the d-dim workspace, refreshed via `reinit_on_pts`
 *          at the lifted points each span).
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parametric dimension of the parent patch.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class BoundaryElementValues
{
public:

    /**
     * @brief Construct the workspace for a (boundary, quadrature) pair.
     *
     * @param boundary           Boundary face this workspace is bound to.
     * @param parent_basis_order Basis derivative order needed by the parent-
     *                           side formulation.
     * @param parent_flags       Quantity flags for the parent-side workspace —
     *                           typically the union of the element's flags and
     *                           the registered boundary fields' flags.
     * @param quadrature         (d-1)-dim quadrature rule on the boundary span.
     *
     * The boundary-side workspace is hardcoded to `basis_order = 1` and
     * `Flags::Deriv1` (the line element `jac` and the boundary tangent `a(0)`
     * are all any boundary integral consumes).
     */
    BoundaryElementValues(const PatchBoundary<T, d>& boundary,
                          Index parent_basis_order, unsigned parent_flags,
                          const QuadratureRule<T, d - 1>& quadrature)
        : boundary_(boundary),
          boundary_vals_(boundary, Index(1), Flags::Deriv1, quadrature),
          parent_vals_(*boundary.parent(), parent_basis_order, parent_flags,
                       quadrature.num_points())
    {}

    /// @brief Number of live (non-zero-volume) spans on the boundary.
    Index num_elements() const { return boundary_vals_.num_elements(); }

    /// @brief Boundary this workspace is bound to.
    const PatchBoundary<T, d>& boundary() const { return boundary_; }

    /// @brief Outward in-surface unit normal n at the current span's quadrature
    ///        points (Q × 3), cached by `reinit`. (d == 2 only.)
    const ColMatrix<T, 3>& outward_normal() const { return outward_normal_; }

    /// @brief In-surface tangent s = a_3 × n at the current span's quadrature
    ///        points (Q × 3), cached by `reinit`. (d == 2 only.)
    const ColMatrix<T, 3>& surface_tangent() const { return surface_tangent_; }

    /**
     * @brief Refresh per-span data for the given live boundary-span index.
     *        Drives the boundary workspace through its standard `reinit`
     *        (maps the boundary quadrature, builds boundary basis + IG),
     *        lifts the resulting parametric points to the parent, and refreshes
     *        the parent workspace at those lifted points via `reinit_on_pts`.
     *
     * @param elem_idx Live boundary-span index.
     */
    void reinit(std::size_t elem_idx)
    {
        // 1. Boundary side: own quadrature mapping + basis + intrinsic geometry
        boundary_vals_.reinit(elem_idx);
        // 2. Lift boundary quadrature points to the parent's parametric space
        const ColMatrix<T, d> lifted =
            boundary_.lift_to_parent(boundary_vals_.mapped_pts_);
        // 3. Look up the parent span containing this boundary span
        //    (d=2 case: boundary is 1D, flat boundary span = span_indices_[0])
        const Index flat_bdy_span = boundary_vals_.span_indices_[0];
        const Index parent_flat_span = boundary_.parent_flat_span(flat_bdy_span);
        // 4. Decode parent flat span to per-direction spans
        const auto parent_spans =
            parent_vals_.patch().decode_span(parent_flat_span);
        // 5. Parent side: basis + intrinsic + extrinsic geometry at lifted points
        parent_vals_.reinit_on_pts(parent_spans, lifted);
        // 6. Cache the boundary frame the projection fields contract against.
        if constexpr (d == 2) {
            outward_normal_ = boundary_.eval_outward_normal(boundary_vals_, parent_vals_);
            const Index Q = static_cast<Index>(outward_normal_.rows());
            surface_tangent_.resize(Q, 3);
            for (Index q = 0; q < Q; ++q) {
                const Vector3<T> a3 = parent_vals_.A3_.row(q).transpose();
                const Vector3<T> n  = outward_normal_.row(q).transpose();
                surface_tangent_.row(q) = a3.cross(n).transpose();
            }
        }
    }

    /**
     * @brief Refresh per-span data at caller-supplied boundary parametric points
     *        within a known live boundary span, instead of self-driving the
     *        boundary quadrature.
     *
     * @details Same pipeline as `reinit` (lift, parent `reinit_on_pts`, frame
     *          cache) but the boundary points come from the caller, not from this
     *          boundary's own quadrature. This is the seam multipatch coupling
     *          uses to evaluate a partner boundary at points coincident with the
     *          driving side's quadrature stations. The supplied points must lie in
     *          `live_boundary_span` — there is no containment check, so the caller
     *          owns parametric correctness (coupling conditions assert physical
     *          coincidence as a guard). A future non-conforming (mortar) driver
     *          reuses this with point-inversion-derived points.
     *
     * @param live_boundary_span Live boundary-span index the points fall in.
     * @param boundary_pts       Boundary parametric points (Q × (d-1)).
     */
    void reinit_on_boundary_pts(Index live_boundary_span,
                                const ColMatrix<T, d - 1>& boundary_pts)
    {
        // 1. Boundary side: basis + intrinsic geometry at the supplied points.
        const auto bdy_spans =
            boundary_vals_.patch().tensor_product().decode_element(live_boundary_span);
        boundary_vals_.reinit_on_pts(bdy_spans, boundary_pts);
        // 2. Lift the supplied boundary points to the parent's parametric space.
        const ColMatrix<T, d> lifted = boundary_.lift_to_parent(boundary_pts);
        // 3. Parent span containing this boundary span.
        const Index parent_flat_span = boundary_.parent_flat_span(bdy_spans[0]);
        const auto parent_spans =
            parent_vals_.patch().decode_span(parent_flat_span);
        // 4. Parent side: basis + intrinsic + extrinsic geometry at lifted points.
        parent_vals_.reinit_on_pts(parent_spans, lifted);
        // 5. Cache the boundary frame the projection fields contract against.
        if constexpr (d == 2) {
            outward_normal_ = boundary_.eval_outward_normal(boundary_vals_, parent_vals_);
            const Index Q = static_cast<Index>(outward_normal_.rows());
            surface_tangent_.resize(Q, 3);
            for (Index q = 0; q < Q; ++q) {
                const Vector3<T> a3 = parent_vals_.A3_.row(q).transpose();
                const Vector3<T> n  = outward_normal_.row(q).transpose();
                surface_tangent_.row(q) = a3.cross(n).transpose();
            }
        }
    }

    ElementValues<T, d - 1> boundary_vals_;

    ElementValues<T, d>     parent_vals_;

private:

    const PatchBoundary<T, d>& boundary_;

    /// @brief Cached outward in-surface unit normal n (Q × 3), refreshed per span.
    ColMatrix<T, 3> outward_normal_;

    /// @brief Cached in-surface tangent s = a_3 × n (Q × 3), refreshed per span.
    ColMatrix<T, 3> surface_tangent_;
};

} // namespace pyck

#endif // PYCK_BOUNDARY_ELEMENT_VALUES_HPP
