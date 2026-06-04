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

// === Flag constants =================================================================

/**
 * @brief Per-quantity flag constants for `ElementValues`, forming a gradual
 *        dependency ladder keyed on derivative order.
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
constexpr unsigned NormalDeriv1    = 1u << 6;  ///< A_{3,β} = −B^α_β A_α      (d=2)
constexpr unsigned Connection      = 1u << 7;  ///< Christoffel Γ_{ε,αβ} = A_ε·A_{,αβ}

} // namespace Flags

// === Point-bound primitive views ====================================================
//
// Returned by the verbose `ElementValues` accessors so a kernel binds a geometric
// primitive at a point once, then indexes its components like the notation:
//   const auto A = ev.cov_basis(q);  const Vector3<T> A1 = A(0), A2 = A(1);
//   const auto B = ev.curvature(q);  const T B12 = B(0, 1);
// Each holds only references and the point index — nothing large is copied.

namespace detail
{

/// @brief Indexable view of a single-index 3-vector field packed slot-major in a
///        (slots·Q)×3 buffer: `(s) → row(s·Q + q)`. Serves the covariant basis A_α
///        and the Weingarten derivative A_{3,β}.
template <std::floating_point T>
struct VectorFieldView
{
    const ColMatrix<T, 3>& buf;
    Index                  Q;
    Index                  q;
    Vector3<T> operator()(Index s) const { return buf.row(s * Q + q).transpose(); }
};

/// @brief Indexable view of a symmetric two-index 3-vector field A_{ij} (e.g. A_{α,β}).
template <std::floating_point T, std::size_t d>
struct VectorField2View
{
    const ColMatrix<T, 3>& buf;
    Index                  Q;
    Index                  q;
    Vector3<T> operator()(Index i, Index j) const
    { return buf.row(pack2<d>(i, j) * Q + q).transpose(); }
};

/// @brief Indexable view of a symmetric three-index 3-vector field A_{ijk} (A_{α,βγ}).
template <std::floating_point T, std::size_t d>
struct VectorField3View
{
    const ColMatrix<T, 3>& buf;
    Index                  Q;
    Index                  q;
    Vector3<T> operator()(Index i, Index j, Index k) const
    { return buf.row(pack3<d>(i, j, k) * Q + q).transpose(); }
};

/// @brief Indexable view of the second-kind Christoffel Γ^k_{ij} at a point, raised
///        on demand from the cached first kind. `(k, i, j) → scalar`; sym in (i, j).
template <std::floating_point T, std::size_t d>
struct ChristoffelView
{
    const Matrix<T>& metric_inv;
    const Matrix<T>& christoffel_first;
    Index            q;
    T operator()(Index k, Index i, Index j) const
    {
        constexpr Index n_d2 = Index(d * (d + 1) / 2);
        const Index p = pack2<d>(i, j);
        T g = T(0);
        for (Index e = 0; e < Index(d); ++e)
            g += metric_inv(q, pack2<d>(k, e)) * christoffel_first(q, e * n_d2 + p);
        return g;
    }
};

/// @brief Indexable view of shape values at a fixed point: `(i) → N^i`.
template <std::floating_point T>
struct ShapeView
{
    const Matrix<T>& buf;   ///< basis_derivs[0], shape (N, Q).
    Index            q;
    T operator()(Index i) const { return buf(i, q); }
};

/// @brief Indexable view of shape gradients at a fixed point: `(i, α) → N^i_{,α}`.
template <std::floating_point T, std::size_t d>
struct ShapeGradView
{
    const Matrix<T>& buf;   ///< basis_derivs[1], shape (N·d, Q).
    Index            q;
    T operator()(Index i, Index a) const { return buf(i * Index(d) + a, q); }
};

} // namespace detail

// === ElementValues ==================================================================

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

    /// @brief Bulk constructor.
    ElementValues(const Patch<T, d>& patch, Index basis_order, unsigned flags,
                  const QuadratureRule<T, d>& quadrature)
        : ElementValues(patch, basis_order, flags, quadrature.num_points(), &quadrature)
    {}

    /// @brief External-points constructor.
    ElementValues(const Patch<T, d>& patch, Index basis_order, unsigned flags,
                  std::size_t Q)
        : ElementValues(patch, basis_order, flags, Q, nullptr)
    {}

    // === Reinit =====================================================================

    /**
     * @brief Recompute values at the given element.
     * 
     * @param elem_idx The index of the element where to compute the values.
     */
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
    ///        in a known span.
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

        // Basis and derivatives
        geometry::intrinsic::compute_position_derivatives<T, d>(basis_derivs, act_pts_, position_derivs, 
                                                                required_position_order(flags_));

        // Geometric primitive quantities (gated by flags)
        if (flags_ & Flags::Deriv1)
            geometry::intrinsic::compute_metric<T, d>(position_derivs,
                                                      metric_, metric_inv_, jac);
        if (flags_ & Flags::Connection)
            geometry::intrinsic::compute_christoffel_first<T, d>(position_derivs,
                                                                 christoffel_first);

        if constexpr (d == 2) {
            if (flags_ & Flags::Normal)
                geometry::surface::compute_normal<T>(position_derivs, jac, A3_);
            if (flags_ & Flags::Curvature)
                geometry::surface::compute_curvature<T>(position_derivs, A3_, curvature_);
            if (flags_ & Flags::NormalDeriv1)
                geometry::surface::compute_normal_derivative<T>(position_derivs, curvature_,
                                                                metric_inv_, A3_d_);
        }
    }

    // === Properties =================================================================

    /// @brief Number of live (non-zero-volume) elements in the patch.
    Index num_elements() const { return patch_.tensor_product().num_elements(); }

    /// @brief Patch this workspace is bound to.
    const Patch<T, d>& patch() const { return patch_; }

    /// @brief Basis derivative order this workspace was sized for.
    Index basis_order() const { return basis_order_; }

    /// @brief Quantity flag bitmask this workspace was constructed with.
    unsigned flags() const { return flags_; }

    // === Indices ====================================================================

    /// @brief Per-direction knot-span indices.
    std::array<Index, d> span_indices_;

    /// @brief Active control-point indices for the current element.
    std::vector<Index> elem_cps_;

    /// @brief Gathered active control-point coordinates.
    ColMatrix<T, 3> act_pts_;

    /// @brief Global DOF indices for the current element.
    std::vector<Index> elem_dofs_;

    // === Basis ======================================================================

    /// @brief Per-direction univariate basis values and derivatives.
    std::array<std::vector<Matrix<T>>, d> uni_basis_derivs;

    /// @brief Basis values and derivatives.
    std::vector<Matrix<T>> basis_derivs;

    // === Quadrature =================================================================

    /// @brief Quadrature points in parametric coordinates.
    ColMatrix<T, d> mapped_pts_;

    /// @brief Quadrature weights for the current element.
    Vector<T> mapped_weights_;

    // === Intrinsic geometry storage =================================================
    //
    // Raw packed buffers. Kernels should prefer the per-quadrature-point accessors
    // below (`metric(q)`, `A(q, i)`, …); these are kept public for the operators and
    // closed-form geometry tests that read whole buffers or hot packed components.

    /// @brief Per-order packed position-derivative storage, populated up to the
    ///        highest derivative order the flags request (`required_position_order`).
    std::vector<ColMatrix<T, 3>> position_derivs;

    /// @brief Packed covariant metric A_{αβ}.
    Matrix<T> metric_;

    /// @brief Packed contravariant metric A^{αβ}.
    Matrix<T> metric_inv_;

    /// @brief Jacobian √det A, length Q.
    Vector<T> jac;

    /// @brief Christoffel symbols of the first kind Γ_{ε,(αβ)} = A_ε·A_{,αβ}-
    Matrix<T> christoffel_first;

    // === Extrinsic geometry storage (meaningful for d == 2 only) =====================

    /// @brief Unit surface normal A_3, shape Q × 3.
    ColMatrix<T, 3> A3_;

    /// @brief First derivatives of the unit normal A_{3,β} (Weingarten), packed
    ///        β-major: A_{3,β} at q in row β·Q + q. Shape (d·Q) × 3.
    ColMatrix<T, 3> A3_d_;

    // === Per-point accessors ========================================================
    //
    // One rule across the surface: bind the quadrature point with `q`, then index
    // everything else on the returned primitive. Geometry carries no basis index
    //   const auto A = ev.cov_basis(q);   const Vector3<T> A1 = A(0), A2 = A(1);
    //   const auto B = ev.curvature(q);   const T B12 = B(0, 1);
    // while shape functions carry the basis index first
    //   const auto G = ev.dN(q);          const T G1i = G(0, 0), G2i = G(0, 1);
    // which mirrors how the operators read (`hess(i, α, β)`) rather than deal.II's flat
    // `shape_grad(i, q)`. Symmetric 2-tensors also expose a packed-scalar form for the
    // operators' hot path and the closed-form geometry tests.

    /// @brief Number of quadrature / evaluation points this workspace holds.
    Index num_points() const { return static_cast<Index>(basis_derivs[0].cols()); }

    // --- Basis (shape functions): bind at q, then index basis i (+ direction) -------

    /// @brief Shape values at point q; read basis function i as `N(q)(i)`.
    detail::ShapeView<T> N(Index q) const { return {basis_derivs[0], q}; }

    /// @brief Shape gradients N^i_{,α} at point q (metric-normalised);
    ///        read as `dN(q)(i, α)`.
    detail::ShapeGradView<T, d> dN(Index q) const { return {basis_derivs[1], q}; }

    // --- Intrinsic geometry: bind at q, then index components -----------------------

    /// @brief Reference position X(ξ) at point q.
    Vector3<T> position(Index q) const { return pos_row(0, 0, q); }

    /// @brief Covariant basis A_α = ∂_α X at point q; read a tangent as `A(α)`.
    detail::VectorFieldView<T> cov_basis(Index q) const
    { return {position_derivs[1], num_points(), q}; }

    /// @brief First derivatives A_{α,β} of the covariant basis; read as `A_d(α, β)`.
    detail::VectorField2View<T, d> cov_basis_d(Index q) const
    { return {position_derivs[2], num_points(), q}; }

    /// @brief Second derivatives A_{α,βγ} of the covariant basis; read as `A_dd(α, β, γ)`.
    detail::VectorField3View<T, d> cov_basis_dd(Index q) const
    { return {position_derivs[3], num_points(), q}; }

    /// @brief Covariant metric component A_{ij} at point q (packed; hot path / tests).
    T metric(Index q, Index packed) const { return metric_(q, packed); }

    /// @brief Contravariant metric component A^{ij} at point q (packed).
    T metric_inv(Index q, Index packed) const { return metric_inv_(q, packed); }

    /// @brief Covariant metric A_{ij} at point q as an indexable d×d tensor.
    Eigen::Matrix<T, d, d> metric(Index q) const { return sym_tensor(metric_, q); }

    /// @brief Contravariant metric A^{ij} at point q as an indexable d×d tensor.
    Eigen::Matrix<T, d, d> metric_inv(Index q) const { return sym_tensor(metric_inv_, q); }

    /// @brief Inverse metric at point q in the material's Voigt order — upper-triangular
    ///        lexicographic (d=2: A¹¹, A¹², A²²), as `PlaneStress2d`'s `*_voigt` reads.
    Eigen::Matrix<T, d *(d + 1) / 2, 1> metric_inv_voigt(Index q) const
    {
        Eigen::Matrix<T, d *(d + 1) / 2, 1> v;
        Index flat = 0;
        for (Index i = 0; i < Index(d); ++i)
            for (Index j = i; j < Index(d); ++j)
                v(flat++) = metric_inv_(q, pack2<d>(i, j));
        return v;
    }

    /// @brief Second-kind Christoffel Γ^k_{ij} at point q (`Flags::Connection`), raised
    ///        on demand from the cached first kind; read as `Gamma(k, i, j)`.
    detail::ChristoffelView<T, d> christoffel(Index q) const
    { return {metric_inv_, christoffel_first, q}; }

    // --- Extrinsic geometry (d == 2) ------------------------------------------------

    /// @brief Unit surface normal A_3 at point q.
    Vector3<T> normal(Index q) const { return A3_.row(q).transpose(); }

    /// @brief Weingarten derivatives A_{3,β} of the unit normal; read as `nd(β)`.
    detail::VectorFieldView<T> normal_deriv(Index q) const
    { return {A3_d_, num_points(), q}; }

    /// @brief Second fundamental form component b_{ij} at point q (packed).
    T curvature(Index q, Index packed) const { return curvature_(q, packed); }

    /// @brief Second fundamental form b_{ij} at point q as an indexable d×d tensor.
    Eigen::Matrix<T, d, d> curvature(Index q) const { return sym_tensor(curvature_, q); }

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
        if (f & Flags::Connection)       f |= Flags::Deriv2;
        if (f & Flags::NormalDeriv1)     f |= Flags::Curvature;
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

    /// @brief Row q of packed-position @p packed within order-@p ord storage, as a vector.
    Vector3<T> pos_row(Index ord, Index packed, Index q) const
    {
        const Index Q_ = static_cast<Index>(position_derivs[0].rows());
        return position_derivs[ord].row(packed * Q_ + q).transpose();
    }

    /// @brief Expand row q of a diagonals-first packed symmetric buffer to a d×d tensor.
    Eigen::Matrix<T, d, d> sym_tensor(const Matrix<T>& packed, Index q) const
    {
        Eigen::Matrix<T, d, d> M;
        for (Index i = 0; i < Index(d); ++i)
            for (Index j = 0; j < Index(d); ++j)
                M(i, j) = packed(q, pack2<d>(i, j));
        return M;
    }

    /// @brief Second fundamental form storage; exposed via `curvature()`.
    Matrix<T> curvature_;

    const Patch<T, d>&          patch_;
    const QuadratureRule<T, d>* quadrature_;
    Index                       basis_order_;
    unsigned                    flags_;
};

} // namespace pyck

#endif // PYCK_ELEMENT_VALUES_HPP
