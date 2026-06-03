#ifndef PYCK_ELEMENT_VALUES_HPP
#define PYCK_ELEMENT_VALUES_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../basis/tensor_product.hpp"
#include "../quadrature/quadrature.hpp"
#include "../geometry/primitives_intrinsic.hpp"
#include "../geometry/primitives_surface.hpp"
#include "../geometry/patch.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

// === Flag constants (plain unsigned bitmask) =======================================

/**
 * @brief Per-quantity flag constants for `ElementValues`, forming a gradual
 *        dependency ladder keyed on derivative order. A flag auto-triggers the
 *        lower rungs it needs (see `expand_flags`), so a consumer lists only the
 *        top quantities it consumes. Values are plain `unsigned`, combined with
 *        native bitwise ops: `Flags::Normal | Flags::Curvature`.
 */
namespace Flags
{
    constexpr unsigned None            = 0;
    constexpr unsigned Values          = 1u << 0;  ///< R = x(ξ)                  (position order 0)
    constexpr unsigned Deriv1          = 1u << 1;  ///< A_α, metric + inv, jac    (position order 1)
    constexpr unsigned Normal          = 1u << 2;  ///< A_3                       (d=2)
    constexpr unsigned Deriv2          = 1u << 3;  ///< A_{α,β}                   (position order 2)
    constexpr unsigned Curvature       = 1u << 4;  ///< B_{αβ} = A_{α,β}·A_3      (d=2)
    constexpr unsigned Deriv3          = 1u << 5;  ///< A_{α,βγ}                  (position order 3)
} // namespace Flags

// === ElementValues ==================================================================

/**
 * @brief Per-element evaluation workspace for a Patch. Owns basis evaluation
 *        scratch + position derivatives + selected intrinsic / extrinsic
 *        geometric quantities, all at the current element's quadrature points.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parametric dimension of the patch.
 */
template <std::floating_point T, std::size_t d>
class ElementValues
{
public:

    /// @brief Bulk constructor: bound to a quadrature rule. Use with
    ///        `reinit(elem_idx)` to iterate the patch's own elements.
    ElementValues(const Patch<T, d>& patch, Index basis_order, unsigned flags,
                  const QuadratureRule<T, d>& quadrature)
        : ElementValues(patch, basis_order, flags, quadrature.num_points(), &quadrature)
    {}

    /// @brief External-points constructor: sized for @p Q quadrature points
    ///        without a bound QuadratureRule. `reinit(elem_idx)` is not usable;
    ///        callers drive refresh via `reinit_on_pts(spans, pts)`.
    ElementValues(const Patch<T, d>& patch, Index basis_order, unsigned flags,
                  std::size_t Q)
        : ElementValues(patch, basis_order, flags, Q, nullptr)
    {}

    /// @brief Number of live (non-zero-volume) elements in the patch.
    Index num_elements() const { return patch_.tensor_product().num_elements(); }

    /// @brief Patch this workspace is bound to.
    const Patch<T, d>& patch() const { return patch_; }

    /// @brief Basis derivative order this workspace was sized for.
    Index basis_order() const { return basis_order_; }

    /// @brief Quantity flag bitmask this workspace was constructed with.
    unsigned flags() const { return flags_; }

    /// @brief Standard reinit: decode @p elem_idx into per-direction spans,
    ///        map the bound quadrature into the span, then delegate to
    ///        `reinit_on_pts`.
    void reinit(std::size_t elem_idx)
    {
        const auto spans = patch_.tensor_product().decode_element(
            static_cast<Index>(elem_idx));
        std::array<T, d> u_a, u_b;
        for (std::size_t i = 0; i < d; ++i) {
            auto [lo, hi] = patch_.basis(i).knot_vector().span_bounds(spans[i]);
            u_a[i] = lo;
            u_b[i] = hi;
        }
        quadrature_->map_to_domain(u_a, u_b, mapped_pts_, mapped_weights_);
        reinit_on_pts(spans, mapped_pts_);
    }

    /// @brief Refresh basis + geometry at caller-provided parametric points
    ///        in a known span. Does not touch `mapped_pts_` / `mapped_weights_`.
    void reinit_on_pts(const std::array<Index, d>& spans,
                       const ColMatrix<T, d>& pts)
    {
        span_indices_ = spans;

        // Basis evaluation
        patch_.tensor_product().eval_on_span(pts, spans, basis_order_,
                                             uni_basis_derivs, basis_derivs);

        // Active control points
        patch_.dof_mapper().get_element_cps(spans, elem_cps_);
        const auto& all_cps = patch_.control_pts();
        for (Index i = 0; i < static_cast<Index>(elem_cps_.size()); ++i) {
            act_pts_.row(i) = all_cps.row(elem_cps_[i]);
        }

        // Geometric quantities (gated by the flag ladder)
        geometry::intrinsic::compute_position_derivatives<T, d>(
            basis_derivs, act_pts_, position_derivs, required_position_order(flags_));
        if (flags_ & Flags::Deriv1)
            geometry::intrinsic::compute_metric<T, d>(position_derivs, metric, metric_inv, jac);

        if constexpr (d == 2) {
            if (flags_ & Flags::Normal)
                geometry::surface::compute_normal<T>(position_derivs, jac, n);
            if (flags_ & Flags::Curvature)
                geometry::surface::compute_curvature<T>(position_derivs, n, b);
        }
    }

    // === Indices ====================================================================

    /// @brief Per-direction knot-span indices of the most recent reinit.
    std::array<Index, d> span_indices_;

    /// @brief Active control-point indices for the current element, in
    ///        tensor-product column order (matches `basis_derivs` row ordering).
    std::vector<Index> elem_cps_;

    /// @brief Gathered active control-point coordinates, shape `(p+1)^d × 3`.
    ColMatrix<T, 3> act_pts_;

    /// @brief Global DOF indices for the current element. Filled externally by
    ///        `DofLayout::scatter_primal`.
    std::vector<Index> elem_dofs_;

    // === Basis ======================================================================

    /// @brief Per-direction univariate basis values + derivatives at the
    ///        current element's quadrature points.
    std::array<std::vector<Matrix<T>>, d> uni_basis_derivs;

    /// @brief Basis values and derivatives at the current element's quadrature
    ///        points, per-order packed (TensorProduct::eval_on_span layout).
    std::vector<Matrix<T>> basis_derivs;

    // === Quadrature =================================================================

    /// @brief Quadrature points in parametric coordinates for the current
    ///        element, shape (Q, d). Filled by `reinit(elem_idx)`; stale on
    ///        the `reinit_on_pts` path.
    ColMatrix<T, d> mapped_pts_;

    /// @brief Quadrature weights for the current element (length Q). Filled
    ///        by `reinit(elem_idx)`; stale on `reinit_on_pts`.
    Vector<T> mapped_weights_;

    // === Intrinsic geometry storage =================================================
    //
    // Layout: see primitives_intrinsic.hpp for packing conventions.

    /// @brief Per-order packed position-derivative storage, populated up to the
    ///        highest derivative order the flags request (`required_position_order`).
    std::vector<ColMatrix<T, 3>> position_derivs;

    /// @brief Packed covariant metric A_{αβ}.
    Matrix<T> metric;

    /// @brief Packed contravariant metric A^{αβ}.
    Matrix<T> metric_inv;

    /// @brief Jacobian √det A, length Q.
    Vector<T> jac;

    // === Extrinsic geometry storage (meaningful for d == 2 only) =====================

    /// @brief Unit surface normal a_3, shape Q × 3.
    ColMatrix<T, 3> n;

    /// @brief Second fundamental form b_{αβ}.
    Matrix<T> b;

    // === Position-derivative accessors ==============================================

    /// @brief Position x(ξ).
    auto x() const                              { return view_pos(0, 0); }

    /// @brief Covariant basis a_α = ∂_α x.
    auto a(Index i) const                       { return view_pos(1, i); }

    /// @brief Symmetric second derivative a_{αβ} = ∂_{αβ} x.
    auto a_d1(Index i, Index j) const           { return view_pos(2, pack2<d>(i, j)); }

    /// @brief Symmetric third derivative a_{αβγ} = ∂_{αβγ} x.
    auto a_d2(Index i, Index j, Index k) const  { return view_pos(3, pack3<d>(i, j, k)); }

    // The packed metric / inverse-metric / curvature buffers (`metric`, `metric_inv`,
    // `b`) are public Matrices, shape (Q, d(d+1)/2). Index a component at quadrature
    // point q as `ev.metric_inv(q, pack2<d>(α,β))` (a single value) or
    // `ev.metric_inv.col(pack2<d>(α,β))` (the per-q column).

    /// @brief The inverse metric at q in the material's Voigt order — upper-triangular
    ///        lexicographic (for d=2: A¹¹, A¹², A²²). NOTE this REORDERS: the `metric_inv`
    ///        buffer is packed diagonals-first (`pack2`: A¹¹, A²², A¹²), which is not the
    ///        order `PlaneStress2d`'s `*_voigt` reads — so it is an adapter, not a row copy.
    Eigen::Matrix<T, d * (d + 1) / 2, 1> metric_inv_voigt(Index q) const
    {
        Eigen::Matrix<T, d * (d + 1) / 2, 1> v;
        std::size_t flat = 0;
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j)
                v(flat++) = metric_inv(q, pack2<d>(static_cast<Index>(i), static_cast<Index>(j)));
        return v;
    }

private:

    /// @brief Expand requested flags down the dependency ladder so a consumer
    ///        lists only the top quantities it reads. Each rung pulls in the lower
    ///        rungs it needs; one top-to-bottom pass settles. The operators are
    ///        basis-direct (connections built per qp from base vectors), so requesting
    ///        an operator reduces to requesting the position derivatives it reads —
    ///        `Deriv2` for the Hessian / Laplace–Beltrami / vector gradient, `Deriv3`
    ///        for the Laplace–Beltrami gradient.
    static unsigned expand_flags(unsigned f)
    {
        if (f & Flags::Curvature)        f |= Flags::Deriv2 | Flags::Normal;
        if (f & Flags::Deriv3)           f |= Flags::Deriv2;
        if (f & Flags::Deriv2)           f |= Flags::Deriv1;
        if (f & Flags::Normal)           f |= Flags::Deriv1;
        if (f & Flags::Deriv1)           f |= Flags::Values;
        return f;
    }

    /// @brief Highest position-derivative order implied by the (expanded) flags.
    static Index required_position_order(unsigned f)
    {
        if (f & Flags::Deriv3) return 3;
        if (f & Flags::Deriv2) return 2;
        if (f & Flags::Deriv1) return 1;
        return 0;
    }

    /// @brief Shared init helper used by both public constructors.
    ElementValues(const Patch<T, d>& patch, Index basis_order, unsigned flags,
                  std::size_t Q, const QuadratureRule<T, d>* quadrature)
        : patch_(patch), quadrature_(quadrature),
          basis_order_(basis_order), flags_(expand_flags(flags))
    {
        // The basis must be evaluated deep enough for the position-derivative
        // orders the flags request (kernels read up to order 3).
        assert(basis_order_ >= required_position_order(flags_)
               && "ElementValues: basis_order too low for requested flags");

        const Index Qi = static_cast<Index>(Q);
        mapped_pts_.resize(Qi, static_cast<Index>(d));
        mapped_weights_.resize(Qi);

        Index N = 1;
        for (std::size_t i = 0; i < d; ++i) {
            const Index Ni = static_cast<Index>(patch.basis(i).degree()) + 1;
            N *= Ni;

            uni_basis_derivs[i].resize(basis_order + 1);
            for (Index k = 0; k <= basis_order; ++k) {
                uni_basis_derivs[i][k].resize(Ni, Qi);
            }
        }
        resize_basis_buffer<T, d>(basis_derivs, N, Qi, basis_order);

        elem_cps_.reserve(static_cast<std::size_t>(N));
        act_pts_.resize(N, 3);
    }

    /// @brief Build a Q × 3 view onto packed-position @p packed within
    ///        order-@p ord storage.
    auto view_pos(Index ord, Index packed) const
    {
        const Index Q_ = static_cast<Index>(position_derivs[0].rows());
        return position_derivs[ord].middleRows(packed * Q_, Q_);
    }

    const Patch<T, d>&          patch_;
    const QuadratureRule<T, d>* quadrature_;
    Index                       basis_order_;
    unsigned                    flags_;
};

} // namespace pyck

#endif // PYCK_ELEMENT_VALUES_HPP
