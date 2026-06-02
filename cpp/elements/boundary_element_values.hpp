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
    }

    ElementValues<T, d - 1> boundary_vals_;

    ElementValues<T, d>     parent_vals_;

private:

    const PatchBoundary<T, d>& boundary_;
};

} // namespace pyck

#endif // PYCK_BOUNDARY_ELEMENT_VALUES_HPP
