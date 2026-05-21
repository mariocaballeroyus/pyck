#ifndef PYCK_VALUES_HPP
#define PYCK_VALUES_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../basis/bspline_algorithms.hpp"
#include "../basis/tensor_product.hpp"
#include "../quadrature/quadrature.hpp"
#include "intrinsic_geometry.hpp"
#include "patch.hpp"
#include "../types.hpp"

// TODO: re-introduce ExtrinsicGeometry<T, d> as a member of PatchValues /
// BoundaryPatchValues (parent side) once we resolve the d == 2 constraint —
// either via specialization, [[no_unique_address]] with a placeholder for
// d != 2, or by lifting the requires clause off ExtrinsicGeometry itself.
// Removed for now so PatchValues<T, 1> (beams) instantiates cleanly.

namespace pyck
{

/**
 * @brief Per-element evaluation workspace for a Patch.
 *
 * @details Holds the basis-layer recurrence scratch, per-order packed basis
 *          values, mapped quadrature data, and intrinsic geometry for one
 *          element. A future reinit(elem_idx) will populate these by mapping
 *          the element's quadrature points and calling the tensor-product
 *          basis evaluator. ExtrinsicGeometry will be added back via
 *          conditional inclusion (see file header TODO).
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parametric dimension of the patch.
 */
template <std::floating_point T, std::size_t d>
class PatchValues
{
public:

    /**
     * @brief Construct the workspace for a (patch, quadrature) pair, evaluating
     *        basis derivatives up to @p order.
     *
     * @details Stores const references to the patch and quadrature rule, and
     *          caches the derivative order requested by the consumer (typically
     *          `element.min_order()`). Sizes every per-element scratch buffer
     *          to one element's worth: mapped_pts (Q × d), mapped_weights (Q),
     *          bspline_scratch (max N × order), basis_out (per-order packed
     *          buffer for N = ∏(p_i+1), Q points, derivative order). Subsequent
     *          reinit() calls reuse these buffers in place — no allocation per
     *          element.
     *
     *          IntrinsicGeometry is left default-constructed for now; it will
     *          be bound during reinit() once IntrinsicGeometry grows its own
     *          reinit method.
     */
    PatchValues(const Patch<T, d>&          patch,
                Index                       order,
                const QuadratureRule<T, d>& quadrature)
        : patch_(patch), quadrature_(quadrature), order_(order)
    {
        const Index Q = static_cast<Index>(quadrature.num_points());
        mapped_pts.resize(Q, static_cast<Index>(d));
        mapped_weights.resize(Q);

        // N = ∏(p_i + 1) active basis functions per element; N_max = the
        // largest 1D fan, which is what the Cox-de Boor scratch must hold.
        // Per-direction univariate buffers `uni_scratch[i][k]` are sized
        // (p_i+1) × Q so the 1D `Basis::eval_on_span` calls inside
        // `TensorProduct::eval_on_span` see no-op Eigen resizes. Also cache
        // the per-axis num_spans for fast flat-index recovery in reinit().
        Index N = 1;
        for (std::size_t i = 0; i < d; ++i) {
            const Index Ni = static_cast<Index>(patch.basis(i).degree()) + 1;
            N *= Ni;
            intervals_[i] = static_cast<Index>(patch.basis(i).knot_vector().num_spans());

            uni_scratch[i].resize(order_ + 1);
            for (Index k = 0; k <= order_; ++k) {
                uni_scratch[i][k].resize(Ni, Q);
            }
        }

        resize_basis_buffer<T, d>(basis_out, N, Q, order_);

        elem_nodes.reserve(static_cast<std::size_t>(N));
    }

    /// @brief Number of live (non-zero-volume) elements in the patch.
    Index num_elements() const { return patch_.tensor_product().num_elements(); }

    /// @brief Patch this workspace is bound to.
    const Patch<T, d>& patch() const { return patch_; }

    /**
     * @brief Refresh per-element data for the given live element index.
     *
     * @details `live_idx ∈ [0, num_elements())` enumerates only non-zero-volume
     *          elements (zero-width clamped spans are skipped by construction).
     *          Decodes the live index via `TensorProduct::decode_element` to
     *          recover the per-direction knot-span indices, looks up the
     *          element's parametric bounds, maps the quadrature rule into
     *          `mapped_pts` / `mapped_weights`, then evaluates the
     *          tensor-product basis on this span into `basis_out` (using
     *          `bspline_scratch` as the recurrence workspace).
     *
     *          The flat element index over the full span enumeration is
     *          recovered in `elem_idx` for downstream consumers (DofMapper,
     *          element kernels). IntrinsicGeometry binding will be added once
     *          it grows a .reinit(basis, act_pts) method.
     *
     * @param live_idx Live element index. No zero-volume check is needed.
     */
    void reinit(std::size_t live_idx)
    {
        span_indices_ = patch_.tensor_product().decode_element(
                            static_cast<Index>(live_idx));

        // Flat span index (U-inner), matching DofMapper's convention.
        elem_idx = 0;
        std::size_t stride = 1;
        for (std::size_t i = 0; i < d; ++i) {
            elem_idx += static_cast<std::size_t>(span_indices_[i]) * stride;
            stride  *= static_cast<std::size_t>(intervals_[i]);
        }

        std::array<T, d> u_a, u_b;
        for (std::size_t i = 0; i < d; ++i) {
            auto [lo, hi] = patch_.basis(i).knot_vector().span_bounds(span_indices_[i]);
            u_a[i] = lo;
            u_b[i] = hi;
        }

        quadrature_.map_to_domain(u_a, u_b, mapped_pts, mapped_weights);

        patch_.tensor_product().eval_on_span(mapped_pts, span_indices_, order_,
                                             uni_scratch, basis_out);

        patch_.dof_mapper().get_element_dofs(span_indices_, elem_nodes);
    }

    // === Basis ======================================================================

    /// @brief Per-direction univariate basis values + derivatives at the
    ///        current element's quadrature points. `uni_scratch[dim][k]` is
    ///        the (p_dim+1) × Q matrix of k-th derivatives along direction
    ///        `dim`; tensor-producted by `eval_on_span` into `basis_out`.
    std::array<std::vector<Matrix<T>>, d> uni_scratch;

    /// @brief Basis values and derivatives at the current element's quadrature
    ///        points, per-order packed (TensorProduct::eval_on_span layout).
    std::vector<Matrix<T>> basis_out;

    // === Quadrature =================================================================

    /// @brief Quadrature points in parametric coordinates for the current
    ///        element, shape (Q, d). Filled by QuadratureRule::map_to_domain.
    ColMatrix<T, d> mapped_pts;

    /// @brief Quadrature weights for the current element (length Q), scaled by
    ///        the reference-to-parametric Jacobian.
    Vector<T> mapped_weights;

    // === Geometry ===================================================================

    /// @brief Metric, Jacobians, Christoffel symbols at the current element's
    ///        quadrature points.
    IntrinsicGeometry<T, d> intrinsic_geometry;

    /// @brief Flat element index (over the full span enumeration, including
    ///        zero-width spans) of the most recent reinit(). Stays in sync
    ///        with the DofMapper's element-indexing convention.
    std::size_t elem_idx = static_cast<std::size_t>(-1);

    /// @brief Active control-point indices for the current element, in
    ///        tensor-product column order (matches `basis_out` row ordering).
    std::vector<Index> elem_nodes;

    /// @brief Global DOF indices for the current element (control points
    ///        expanded by the stride of the assembly's primal block). Filled
    ///        externally by `DofLayout::scatter_primal`; lazy-sized on the
    ///        first call, allocation-free thereafter.
    std::vector<Index> elem_dofs;

private:

    const Patch<T, d>&          patch_;
    const QuadratureRule<T, d>& quadrature_;
    Index                       order_;
    std::array<Index, d>        span_indices_;
    std::array<Index, d>        intervals_;
};

/**
 * @brief Per-span evaluation workspace for a PatchBoundary.
 *
 * @details Boundary integration is intrinsically a two-basis problem: the
 *          integrand involves both the boundary's own (d-1)-dim basis (for
 *          multipliers, the boundary measure, normal-traction terms) and
 *          the parent's d-dim basis evaluated at the lifted boundary
 *          quadrature points (because primary unknowns live on the parent).
 *          This workspace holds the boundary side here; the parent-side
 *          basis scratch, output buffer, and lifted points will be added
 *          when reinit() is wired. ExtrinsicGeometry omitted for now (see
 *          file header TODO).
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parametric dimension of the parent patch.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class BoundaryPatchValues
{
public:

    // === Boundary side ((d-1)-dimensional) ==========================================

    /// @brief Boundary basis values and derivatives at the current span's
    ///        quadrature points, per-order packed.
    std::vector<Matrix<T>> basis_out;

    /// @brief Quadrature points in boundary parametric coordinates for the
    ///        current span, shape (Q, d-1). Filled by QuadratureRule::map_to_domain.
    ColMatrix<T, d - 1> mapped_pts;

    /// @brief Boundary quadrature weights (length Q), scaled by the
    ///        reference-to-parametric Jacobian on the boundary span.
    Vector<T> mapped_weights;

    /// @brief Metric and Jacobians on the (d-1)-dimensional boundary.
    IntrinsicGeometry<T, d - 1> intrinsic_geometry;

    // === Parent side (d-dimensional, at lifted boundary quadrature points) ==========
    //
    // Parent-side basis scratch / output / lifted points to be added when
    // reinit() is wired. Parent intrinsic + extrinsic geometry live here now
    // since they're cheap to declare and natural to mirror.

    /// @brief Metric, Jacobians, Christoffel symbols on the parent patch at
    ///        the lifted boundary quadrature points.
    IntrinsicGeometry<T, d> parent_intrinsic_geometry;
};

} // namespace pyck

#endif // PYCK_VALUES_HPP
