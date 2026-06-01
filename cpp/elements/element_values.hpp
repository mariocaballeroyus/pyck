#ifndef PYCK_ELEMENT_VALUES_HPP
#define PYCK_ELEMENT_VALUES_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../basis/tensor_product.hpp"
#include "../quadrature/quadrature.hpp"
#include "../geometry/intrinsic_geometry.hpp"
#include "../geometry/surface_geometry.hpp"
#include "../geometry/element_kernels.hpp"
#include "../geometry/patch.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

// === Flag constants (plain unsigned bitmask) =======================================

/**
 * @brief Per-quantity flag constants for `ElementValues`. Independent bits
 *        select which derived geometric quantities the workspace computes
 *        during `reinit`. Values are plain `unsigned`, combined with native
 *        bitwise ops: `Flags::Metric | Flags::Christoffels`.
 */
namespace Flags
{
    constexpr unsigned None           = 0;
    constexpr unsigned Metric         = 1u << 0;  ///< g, g_inv, jac
    constexpr unsigned Christoffels   = 1u << 1;  ///< Γ^k_{ij}
    constexpr unsigned ChristoffelsD1 = 1u << 2;  ///< ∂_γ Γ^k_{ij}
    constexpr unsigned Normal         = 1u << 3;  ///< a_3           (d=2)
    constexpr unsigned NormalD1       = 1u << 4;  ///< ∂_β a_3       (d=2)
    constexpr unsigned Curvature      = 1u << 5;  ///< b_{αβ}, b^α_β (d=2)
    constexpr unsigned KernelHessian   = 1u << 6;  ///< H_{iαβ}   (per-basis covariant Hessian)
    constexpr unsigned KernelLB        = 1u << 7;  ///< L_i       (per-basis Laplace–Beltrami)
    constexpr unsigned KernelVectorGrad= 1u << 8;  ///< D_{iλαβ}  (per-basis covariant vector gradient)
    constexpr unsigned KernelLBGradient= 1u << 9;  ///< P_{iα}    (per-basis Laplace–Beltrami gradient)
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

        // Geometric quantities (gated by flags)
        geometry::intrinsic::compute_position_derivatives<T, d>(results_, act_pts_, position_data);
        if (flags_ & Flags::Metric)
            geometry::intrinsic::compute_metric<T, d>(position_data, g_data, g_inv_data, jac);
        if (flags_ & Flags::Christoffels)
            geometry::intrinsic::compute_christoffels<T, d>(position_data, g_inv_data, Gamma_data);
        if (flags_ & Flags::ChristoffelsD1)
            geometry::intrinsic::compute_christoffels_d1<T, d>(position_data, g_inv_data, Gamma_d1_data);
        if constexpr (d == 2) {
            if (flags_ & Flags::Normal)
                geometry::surface::compute_normal<T>(position_data, jac, n);
            if (flags_ & Flags::NormalD1)
                geometry::surface::compute_normal_derivatives<T>(position_data, jac, n, n_d1_data);
            if (flags_ & Flags::Curvature)
                geometry::surface::compute_curvature<T>(position_data, g_inv_data, n, b_data, b_mixed_data);

            // Per-(qp, basis) kernels (surface elements only). Basis-direct:
            // connection formed from position derivatives + g_inv, no Christoffels.
            if (flags_ & Flags::KernelHessian)
                geometry::kernels::compute_kernel_hessian<T, d>(results_, position_data, g_inv_data, H_kernel_data);
            if (flags_ & Flags::KernelLB)
                geometry::kernels::compute_kernel_laplace_beltrami<T, d>(H_kernel_data, g_inv_data, L_kernel_data);
            if (flags_ & Flags::KernelVectorGrad)
                geometry::kernels::compute_kernel_vector_gradient<T, d>(results_, position_data, D_kernel_data);
            if (flags_ & Flags::KernelLBGradient) {
                lb_aux_ = compute_laplace_grad_aux<T, d>(position_data, g_inv_data);
                geometry::kernels::compute_kernel_lb_gradient<T, d>(results_, lb_aux_, g_inv_data, P_kernel_data);
            }
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

    /// @brief Per-order packed position-derivative storage (always populated
    ///        up to `flags_.basis_order`).
    std::vector<ColMatrix<T, 3>> position_data;

    /// @brief Packed covariant metric g_{αβ}.
    Matrix<T> g_data;

    /// @brief Packed contravariant metric g^{αβ}.
    Matrix<T> g_inv_data;

    /// @brief Jacobian √det g, length Q.
    Vector<T> jac;

    /// @brief Packed Christoffel symbols Γ^k_{ij}.
    Matrix<T> Gamma_data;

    /// @brief Packed Christoffel derivatives ∂_γ Γ^k_{ij}.
    Matrix<T> Gamma_d1_data;

    // === Extrinsic geometry storage (meaningful for d == 2 only) =====================

    /// @brief Unit surface normal a_3, shape Q × 3.
    ColMatrix<T, 3> n;

    /// @brief Packed normal derivatives ∂_β a_3.
    ColMatrix<T, 3> n_d1_data;

    /// @brief Second fundamental form b_{αβ}.
    Matrix<T> b_data;

    /// @brief Shape operator b^α_β.
    Matrix<T> b_mixed_data;

    // === Per-(qp, basis) kernel storage =============================================

    /// @brief Covariant Hessian H_{iαβ}, packed (N·n_d2, Q): row i·n_d2 + pack2<d>(α,β).
    Matrix<T> H_kernel_data;

    /// @brief Laplace–Beltrami L_i = g^{αβ}H_{iαβ}, shape (N, Q).
    Matrix<T> L_kernel_data;

    /// @brief Covariant vector gradient D_{iλαβ}, dense (N·d³, Q): i·d³ + (λ·d+α)·d+β.
    Matrix<T> D_kernel_data;

    /// @brief Laplace–Beltrami gradient P_{iα}, shape (N·d, Q): i·d + α.
    Matrix<T> P_kernel_data;

    /// @brief Cached inverse-metric / connection-trace gradients for P_{iα}.
    LaplaceGradAux<T, d> lb_aux_;

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

    // === Christoffel accessors ======================================================

    /// @brief Γ^k_{ij}(q) — Q-length view; symmetric in (i, j).
    auto Gamma(Index k, Index i, Index j) const
    {
        constexpr Index n_d2 = d * (d + 1) / 2;
        return Gamma_data.col(k * n_d2 + pack2<d>(i, j));
    }

    /// @brief ∂_γ Γ^k_{ij}(q) — Q-length view; symmetric in (i, j).
    auto Gamma_d1(Index k, Index i, Index j, Index gam) const
    {
        constexpr Index n_d2 = d * (d + 1) / 2;
        return Gamma_d1_data.col(k * (n_d2 * d) + pack2<d>(i, j) * d + gam);
    }

    // === Extrinsic accessors (d == 2 only meaningful) ===============================

    /// @brief ∂_β a_3 — Q × 3 view.
    auto n_d1(Index beta) const
    {
        const Index Q_ = static_cast<Index>(n.rows());
        return n_d1_data.middleRows(beta * Q_, Q_);
    }

    /// @brief b_{αβ}(q). Symmetric in (α, β).
    auto b(Index alpha, Index beta) const
    { return b_data.col(pack2<2>(alpha, beta)); }

    /// @brief b^α_β(q). Not symmetric.
    auto b_mixed(Index alpha, Index beta) const
    { return b_mixed_data.col(alpha * 2 + beta); }

    // === Kernel accessors ===========================================================
    //
    // `Value` and the covariant scalar gradient are the raw basis buffers:
    // N^i(q) = results_[0].col(q)(i), N^i_{,α}(q) = results_[1].col(q)(i·d + α).

    /// @brief Covariant Hessian H_{iαβ}(q) of basis function i. Symmetric in (α, β).
    T H(Index i, Index alpha, Index beta, Index q) const
    {
        constexpr Index n_d2 = d * (d + 1) / 2;
        return H_kernel_data(i * n_d2 + pack2<d>(alpha, beta), q);
    }

    /// @brief Laplace–Beltrami L_i(q) of basis function i.
    T L(Index i, Index q) const { return L_kernel_data(i, q); }

    /// @brief Covariant vector gradient D_{iλαβ}(q): contributes u_{α|β} = D·u^λ.
    T D(Index i, Index lambda, Index alpha, Index beta, Index q) const
    {
        constexpr Index dd = static_cast<Index>(d);
        return D_kernel_data(i * (dd * dd * dd) + (lambda * dd + alpha) * dd + beta, q);
    }

    /// @brief Laplace–Beltrami gradient P_{iα}(q) of basis function i.
    T P(Index i, Index alpha, Index q) const
    {
        constexpr Index dd = static_cast<Index>(d);
        return P_kernel_data(i * dd + alpha, q);
    }

private:

    /// @brief Expand requested flags with their dependencies so an element only
    ///        lists the kernels it consumes. Kernels are basis-direct: they need
    ///        the contravariant metric (and the Hessian for the L–B contraction),
    ///        but no Christoffel array.
    static unsigned expand_flags(unsigned f)
    {
        if (f & Flags::KernelLB)         f |= Flags::KernelHessian;
        if (f & Flags::KernelHessian)    f |= Flags::Metric;
        if (f & Flags::KernelLBGradient) f |= Flags::Metric;
        // KernelVectorGrad needs only position derivatives (always computed).
        return f;
    }

    /// @brief Shared init helper used by both public constructors.
    ElementValues(const Patch<T, d>& patch, Index basis_order, unsigned flags,
                  std::size_t Q, const QuadratureRule<T, d>* quadrature)
        : patch_(patch), quadrature_(quadrature),
          basis_order_(basis_order), flags_(expand_flags(flags))
    {
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
