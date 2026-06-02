#ifndef PYCK_ELEMENT_VALUES_HPP
#define PYCK_ELEMENT_VALUES_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../basis/tensor_product.hpp"
#include "../quadrature/quadrature.hpp"
#include "../geometry/intrinsic_geometry.hpp"
#include "../geometry/surface_geometry.hpp"
#include "../operators/laplace_beltrami_gradient.hpp"
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
    constexpr unsigned Deriv1          = 1u << 1;  ///< A_α, metric g/g_inv, jac  (position order 1)
    constexpr unsigned Normal          = 1u << 2;  ///< A_3                       (d=2)
    constexpr unsigned Deriv2          = 1u << 3;  ///< A_{α,β}                   (position order 2)
    constexpr unsigned Curvature       = 1u << 4;  ///< B_{αβ} = A_{α,β}·A_3      (d=2)
    constexpr unsigned Deriv3          = 1u << 5;  ///< A_{α,βγ}                  (position order 3)
    constexpr unsigned LaplaceBeltramiGradient = 1u << 6;  ///< cache LaplaceBeltramiGradConn (lb_grad_conn_) for the P operator
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
                                             uni_results_, results_);

        // Active control points
        patch_.dof_mapper().get_element_cps(spans, elem_cps_);
        const auto& all_cps = patch_.control_pts();
        for (Index i = 0; i < static_cast<Index>(elem_cps_.size()); ++i) {
            act_pts_.row(i) = all_cps.row(elem_cps_[i]);
        }

        // Geometric quantities (gated by the flag ladder)
        geometry::intrinsic::compute_position_derivatives<T, d>(
            results_, act_pts_, position_data, required_position_order(flags_));
        if (flags_ & Flags::Deriv1)
            geometry::intrinsic::compute_metric<T, d>(position_data, g_data, g_inv_data, jac);

        if constexpr (d == 2) {
            if (flags_ & Flags::Normal)
                geometry::surface::compute_normal<T>(position_data, jac, n);
            if (flags_ & Flags::Curvature)
                geometry::surface::compute_curvature<T>(position_data, n, b_data);

            // The Hessian / vector-gradient connections are cheap and formed on the
            // fly by their operators; only the expensive 3rd-order Laplace–Beltrami
            // gradient connector is cached here.
            if (flags_ & Flags::LaplaceBeltramiGradient)
                lb_grad_conn_ = operators::compute_laplace_beltrami_grad_conn<T, d>(position_data, g_inv_data);
        }
    }

    // === Indices ====================================================================

    /// @brief Per-direction knot-span indices of the most recent reinit.
    std::array<Index, d> span_indices_;

    /// @brief Active control-point indices for the current element, in
    ///        tensor-product column order (matches `results_` row ordering).
    std::vector<Index> elem_cps_;

    /// @brief Gathered active control-point coordinates, shape `(p+1)^d × 3`.
    ColMatrix<T, 3> act_pts_;

    /// @brief Global DOF indices for the current element. Filled externally by
    ///        `DofLayout::scatter_primal`.
    std::vector<Index> elem_dofs_;

    // === Basis ======================================================================

    /// @brief Per-direction univariate basis values + derivatives at the
    ///        current element's quadrature points.
    std::array<std::vector<Matrix<T>>, d> uni_results_;

    /// @brief Basis values and derivatives at the current element's quadrature
    ///        points, per-order packed (TensorProduct::eval_on_span layout).
    std::vector<Matrix<T>> results_;

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
    // Layout: see intrinsic_geometry.hpp for packing conventions.

    /// @brief Per-order packed position-derivative storage, populated up to the
    ///        highest derivative order the flags request (`required_position_order`).
    std::vector<ColMatrix<T, 3>> position_data;

    /// @brief Packed covariant metric g_{αβ}.
    Matrix<T> g_data;

    /// @brief Packed contravariant metric g^{αβ}.
    Matrix<T> g_inv_data;

    /// @brief Jacobian √det g, length Q.
    Vector<T> jac;

    // === Extrinsic geometry storage (meaningful for d == 2 only) =====================

    /// @brief Unit surface normal a_3, shape Q × 3.
    ColMatrix<T, 3> n;

    /// @brief Second fundamental form b_{αβ}.
    Matrix<T> b_data;

    // === Per-(qp) kernel auxiliary storage ==========================================

    /// @brief Cached inverse-metric / connection-trace gradients for P_{iα} — the
    ///        one genuinely expensive 3rd-order connector. The Hessian /
    ///        vector-gradient connections are cheap and formed on the fly from
    ///        base vectors by the operators (nothing connection-like stored).
    operators::LaplaceBeltramiGradConn<T, d> lb_grad_conn_;

    // === Position-derivative accessors ==============================================

    /// @brief Position x(ξ).
    auto x() const                              { return view_pos(0, 0); }

    /// @brief Covariant basis a_α = ∂_α x.
    auto a(Index i) const                       { return view_pos(1, i); }

    /// @brief Symmetric second derivative a_{αβ} = ∂_{αβ} x.
    auto a_d1(Index i, Index j) const           { return view_pos(2, pack2<d>(i, j)); }

    /// @brief Symmetric third derivative a_{αβγ} = ∂_{αβγ} x.
    auto a_d2(Index i, Index j, Index k) const  { return view_pos(3, pack3<d>(i, j, k)); }

    // === Metric accessors ===========================================================

    /// @brief Covariant metric g_{αβ}(q). Symmetric in (α, β).
    auto g(Index i, Index j) const     { return g_data.col(pack2<d>(i, j)); }

    /// @brief Contravariant metric g^{αβ}(q). Symmetric in (α, β).
    auto g_inv(Index i, Index j) const { return g_inv_data.col(pack2<d>(i, j)); }

    // === Extrinsic accessors (d == 2 only meaningful) ===============================

    /// @brief b_{αβ}(q). Symmetric in (α, β).
    auto b(Index alpha, Index beta) const
    { return b_data.col(pack2<2>(alpha, beta)); }

    // Differential operators (covariant Hessian, Laplace–Beltrami, in-plane vector
    // gradient, Laplace–Beltrami gradient) are NOT exposed here: they live in
    // `cpp/operators/` and are called by elements on the primitives above
    // (`results_`, `position_data`, `g_inv_data`, `lb_grad_conn_`).

private:

    /// @brief Expand requested flags down the dependency ladder so a consumer
    ///        lists only the top quantities it reads. Each rung pulls in the lower
    ///        rungs it needs; one top-to-bottom pass settles. The operators are
    ///        basis-direct (connections formed from base vectors), so requesting an
    ///        operator reduces to requesting the position derivatives it reads —
    ///        `Deriv2` for the Hessian / Laplace–Beltrami / vector gradient. Only
    ///        the Laplace–Beltrami gradient needs more: its cached connector and the
    ///        order-3 derivatives behind it.
    static unsigned expand_flags(unsigned f)
    {
        if (f & Flags::LaplaceBeltramiGradient)  f |= Flags::Deriv3;
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

            uni_results_[i].resize(basis_order + 1);
            for (Index k = 0; k <= basis_order; ++k) {
                uni_results_[i][k].resize(Ni, Qi);
            }
        }
        resize_basis_buffer<T, d>(results_, N, Qi, basis_order);

        elem_cps_.reserve(static_cast<std::size_t>(N));
        act_pts_.resize(N, 3);
    }

    /// @brief Build a Q × 3 view onto packed-position @p packed within
    ///        order-@p ord storage.
    auto view_pos(Index ord, Index packed) const
    {
        const Index Q_ = static_cast<Index>(position_data[0].rows());
        return position_data[ord].middleRows(packed * Q_, Q_);
    }

    const Patch<T, d>&          patch_;
    const QuadratureRule<T, d>* quadrature_;
    Index                       basis_order_;
    unsigned                    flags_;
};

// === Voigt-packed metric helpers ====================================================

/**
 * @brief Pack the upper-triangular metric components into a column vector at
 *        q-point @p q. For d=2 this is (g11, g12, g22)ᵀ; for d=3
 *        (g11, g12, g13, g22, g23, g33)ᵀ.
 */
template <std::floating_point T, std::size_t d>
Eigen::Matrix<T, d * (d + 1) / 2, 1>
g_voigt(const ElementValues<T, d>& ev, Index q)
{
    Eigen::Matrix<T, d * (d + 1) / 2, 1> v;
    std::size_t flat = 0;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = i; j < d; ++j)
            v(flat++) = ev.g(i, j)(q);
    return v;
}

/// @brief Same packing as @ref g_voigt, applied to the inverse metric.
template <std::floating_point T, std::size_t d>
Eigen::Matrix<T, d * (d + 1) / 2, 1>
g_inv_voigt(const ElementValues<T, d>& ev, Index q)
{
    Eigen::Matrix<T, d * (d + 1) / 2, 1> v;
    std::size_t flat = 0;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = i; j < d; ++j)
            v(flat++) = ev.g_inv(i, j)(q);
    return v;
}

} // namespace pyck

#endif // PYCK_ELEMENT_VALUES_HPP
