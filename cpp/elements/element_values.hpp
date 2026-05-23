#ifndef PYCK_ELEMENT_VALUES_HPP
#define PYCK_ELEMENT_VALUES_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../basis/bspline_algorithms.hpp"
#include "../basis/tensor_product.hpp"
#include "../quadrature/quadrature.hpp"
#include "../geometry/extrinsic_geometry.hpp"
#include "../geometry/intrinsic_geometry.hpp"
#include "../geometry/patch.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Per-element evaluation workspace for a Patch.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parametric dimension of the patch.
 */
template <std::floating_point T, std::size_t d>
class ElementValues
{
public:

    /**
     * @brief Construct the workspace for a (patch, quadrature) pair, evaluating
     *        basis derivatives up to @p order.
     */
    ElementValues(const Patch<T, d>& patch, Index order,
                  const QuadratureRule<T, d>& quadrature)
        : patch_(patch), quadrature_(quadrature), order_(order)
    {
        const Index Q = static_cast<Index>(quadrature.num_points());
        mapped_pts_.resize(Q, static_cast<Index>(d));
        mapped_weights_.resize(Q);

        Index N = 1;
        for (std::size_t i = 0; i < d; ++i) {
            const Index Ni = static_cast<Index>(patch.basis(i).degree()) + 1;
            N *= Ni;
            intervals_[i] = static_cast<Index>(patch.basis(i).knot_vector().num_spans());

            uni_results_[i].resize(order_ + 1);
            for (Index k = 0; k <= order_; ++k) {
                uni_results_[i][k].resize(Ni, Q);
            }
        }

        resize_basis_buffer<T, d>(results_, N, Q, order_);

        elem_cps_.reserve(static_cast<std::size_t>(N));
        act_pts_.resize(N, 3);
    }

    /// @brief Number of live (non-zero-volume) elements in the patch.
    Index num_elements() const { return patch_.tensor_product().num_elements(); }

    /// @brief Patch this workspace is bound to.
    const Patch<T, d>& patch() const { return patch_; }

    /**
     * @brief Refresh per-element data for the given live element index.
     *
     * @param elem_idx Live element index. No zero-volume check is needed.
     */
    void reinit(std::size_t elem_idx)
    {
        span_indices_ = patch_.tensor_product().decode_element(static_cast<Index>(elem_idx));

        elem_idx_ = 0;
        std::size_t stride = 1;
        for (std::size_t i = 0; i < d; ++i) {
            elem_idx_ += static_cast<std::size_t>(span_indices_[i]) * stride;
            stride   *= static_cast<std::size_t>(intervals_[i]);
        }

        // Compute span bounds in every direction
        std::array<T, d> u_a, u_b;
        for (std::size_t i = 0; i < d; ++i) {
            auto [lo, hi] = patch_.basis(i).knot_vector().span_bounds(span_indices_[i]);
            u_a[i] = lo;
            u_b[i] = hi;
        }
        // Map quadrature points from gauss to parametric space using span bounds
        quadrature_.map_to_domain(u_a, u_b, mapped_pts_, mapped_weights_);
        // Evaluate tensor product basis in the mapped quadrature points
        patch_.tensor_product().eval_on_span(mapped_pts_, span_indices_, order_,
                                             uni_results_, results_);
        // Get the indices of the element control points
        patch_.dof_mapper().get_element_cps(span_indices_, elem_cps_);
        // Gather active control-point coordinates into the workspace buffer
        const auto& all_cps = patch_.control_pts();
        for (Index i = 0; i < static_cast<Index>(elem_cps_.size()); ++i) {
            act_pts_.row(i) = all_cps.row(elem_cps_[i]);
        }
        // Refresh geometric workspaces from the gathered basis + control points
        intrinsic_geometry_.reinit(results_, act_pts_, order_);
        extrinsic_geometry_.reinit(intrinsic_geometry_);
    }

    // === Indices ====================================================================

    /// @brief Flat element index (over the full span enumeration, including
    ///        zero-width spans) of the most recent reinit(). Stays in sync
    ///        with the DofMapper's element-indexing convention.
    std::size_t elem_idx_ = static_cast<std::size_t>(-1);

    /// @brief Active control-point indices for the current element, in
    ///        tensor-product column order (matches `results_` row ordering).
    std::vector<Index> elem_cps_;

    /// @brief Gathered active control-point coordinates for the current
    ///        element, shape `(p+1)^d × 3`. Row `i` is the 3D coordinate of
    ///        the control point at index `elem_cps_[i]`. Materialised here so
    ///        the inner geometry loop reads contiguous rows instead of paying
    ///        an indirection per access.
    ColMatrix<T, 3> act_pts_;

    /// @brief Global DOF indices for the current element (control points
    ///        expanded by the stride of the assembly's primal block). Filled
    ///        externally by `DofLayout::scatter_primal`; lazy-sized on the
    ///        first call, allocation-free thereafter.
    std::vector<Index> elem_dofs_;

    // === Basis ======================================================================

    /// @brief Per-direction univariate basis values + derivatives at the
    ///        current element's quadrature points. `uni_results_[dim][k]` is
    ///        the (p_dim+1) × Q matrix of k-th derivatives along direction
    ///        `dim`; tensor-producted by `eval_on_span` into `results_`.
    std::array<std::vector<Matrix<T>>, d> uni_results_;

    /// @brief Basis values and derivatives at the current element's quadrature
    ///        points, per-order packed (TensorProduct::eval_on_span layout).
    std::vector<Matrix<T>> results_;

    // === Quadrature =================================================================

    /// @brief Quadrature points in parametric coordinates for the current
    ///        element, shape (Q, d). Filled by QuadratureRule::map_to_domain.
    ColMatrix<T, d> mapped_pts_;

    /// @brief Quadrature weights for the current element (length Q), scaled by
    ///        the reference-to-parametric Jacobian.
    Vector<T> mapped_weights_;

    // === Geometry ===================================================================

    /// @brief Metric, Jacobians, Christoffel symbols at the current element's
    ///        quadrature points.
    IntrinsicGeometry<T, d> intrinsic_geometry_;

    /// @brief Unit normal, normal derivatives and surface curvature at the
    ///        current element's quadrature points. Meaningful only for d == 2
    ///        (empty no-op for other dimensions).
    ExtrinsicGeometry<T, d> extrinsic_geometry_;

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
class BoundaryElementValues
{
public:

    // === Boundary side ((d-1)-dimensional) ==========================================

    /// @brief Boundary basis values and derivatives at the current span's
    ///        quadrature points, per-order packed.
    std::vector<Matrix<T>> results;

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
