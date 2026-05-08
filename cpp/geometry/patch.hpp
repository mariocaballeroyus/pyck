#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <tuple>
#include <vector>
#include <memory>
#include <Eigen/Core>

#include "../basis/basis.hpp"
#include "../basis/tensor.hpp"
#include "../basis/bspline.hpp"
#include "dof_mapper.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
class QuadratureRule;


/**
 * @brief Per-element surface kinematic data: basis derivatives plus all geometric
 *        quantities derived from them.
 *
 * Single source of truth for `Patch<T, 2>` consumers (shell element, covariant
 * shape-derivative correction, future curved-surface boundary conditions). The
 * geometric quantities cost O(Q) to compute given `basis_derivs`; bundling them
 * here lets multiple consumers share a single evaluation.
 *
 * Layout (all matrices have Q rows; K = (p_u+1)*(p_v+1) is active basis count):
 *   - basis_derivs[du*(order+1) + dv] = parametric N^{(du,dv)}, size Q×K
 *   - a1, a2, a3                       — covariant tangents and unit director (Q×3)
 *   - aup1, aup2                       — dual basis a^α = g^{αβ} a_β (Q×3)
 *   - g_inv                            — (g^11, g^12, g^22) (Q×3)
 *   - jac                              — √det g = ||a1×a2|| (Q)
 *   - a11, a12, a22                    — geometry 2nd derivatives (Q×3, order ≥ 2)
 *   - b                                — (b_11, b_12, b_22), b_αβ = a_{αβ}·a3 (Q×3, order ≥ 2)
 *   - christoffel                      — Γ packed as (Γ¹₁₁, Γ¹₁₂, Γ¹₂₂, Γ²₁₁, Γ²₁₂, Γ²₂₂) (Q×6, order ≥ 2)
 *   - a111, a112, a122, a222           — geometry 3rd derivatives (Q×3, order ≥ 3)
 *
 * Fields not produced at the requested order are left default-constructed (size 0).
 */
template <std::floating_point T>
struct SurfaceKinematics
{
    std::vector<Matrix<T>> basis_derivs;

    ColMatrix<T, 3> a1, a2, a3;
    ColMatrix<T, 3> aup1, aup2;
    ColMatrix<T, 3> g_inv;
    Vector<T>       jac;

    ColMatrix<T, 3> a11, a12, a22;
    ColMatrix<T, 3> b;
    ColMatrix<T, 6> christoffel;

    ColMatrix<T, 3> a111, a112, a122, a222;
};


/**
 * @brief Primary template for parametric patches defined by the tensor-product
 *        of univariate basis functions.
 * 
 * This class provides common data and logic for all patches (curves, surfaces, etc.).
 * Mathematical operations like eval_shape_functions are specialized for each dimension.
 */
/**
 * @brief Primary template for parametric patches.
 */
template <std::floating_point T, std::size_t d>
class Patch
{
public:
    /**
     * @brief 1D Constructor (Curve).
     */
    Patch(Ptr<const Basis<T>> basis_u, 
          const ColMatrix<T, 3>& control_pts) requires(d == 1);

    /**
     * @brief 2D Constructor (Surface).
     */
    Patch(Ptr<const Basis<T>> basis_u,
          Ptr<const Basis<T>> basis_v,
          const ColMatrix<T, 3>& control_pts) requires(d == 2);

    virtual ~Patch() = default;

    std::vector<Matrix<T>> eval_basis_functions(const ColMatrix<T, d>& points,
                                                Index span,
                                                std::size_t order = 0) const;
    
                                                std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions(const ColMatrix<T, d>& points,
                         Index span,
                         std::size_t order = 0) const requires(d == 1);
    std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions(const ColMatrix<T, d>& points,
                         Index span,
                         std::size_t order = 0) const requires(d == 2);

    ColMatrix<T, 3> eval_geometry(const ColMatrix<T, d>& points,
                                  Index span) const requires(d == 1);

    ColMatrix<T, 3> eval_geometry(const ColMatrix<T, d>& points,
                                  Index span) const requires(d == 2);

    ColMatrix<T, 3> eval_physical_points(const QuadratureRule<T, d>& quadrature) const requires(d == 1);
    ColMatrix<T, 3> eval_physical_points(const QuadratureRule<T, d>& quadrature) const requires(d == 2);

    /// @brief Get the Greville points of the patch in the parametric domain.
    const ColMatrix<T, d>& greville_points() const { return greville_points_; }

    /// @brief Get the element span containing the Greville point for a global DOF index.
    Index greville_span(Index dof_index) const { return greville_spans_[dof_index]; }

    /**
     * @brief Evaluate shape functions and their derivatives at the Greville point 
     *        corresponding to a global DOF index.
     * 
     * @param dof_index Global DOF index (0 to num_control_pts - 1).
     * @param order Highest order of derivatives to compute.
     * @return Evaluation results (shape function matrices and area element).
     */
    std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions_at_greville(Index dof_index, std::size_t order = 0) const;

    std::array<ColMatrix<T, 3>, d> eval_tangent(
        const std::vector<Matrix<T>>& N,
        const ColMatrix<T, 3>& act_pts) const requires(d == 1);

    std::array<ColMatrix<T, 3>, d> eval_tangent(
        const std::vector<Matrix<T>>& N,
        const ColMatrix<T, 3>& act_pts) const requires(d == 2);

    /**
     * @brief Evaluate the local covariant frame {a1, a2, a3} at a set of
     *        parametric points within a single span.
     *
     * Returns (a1, a2, a3, jac):
     *   - a1, a2: raw covariant tangents a_α = ∂x/∂ξ^α (Q×3, NOT
     *             unit-normalised; their magnitudes are √g_αα).
     *   - a3:     unit director a3 = (a1×a2)/||a1×a2|| (Q×3).
     *   - jac:    area element ||a1×a2|| = √det(g) (Q).
     *
     * Boundary callers should delegate the surface normal a3 to the parent
     * patch via this method rather than recomputing it locally.
     *
     * @note BoundaryField subclasses currently consume the boundary outward
     * normal as (n_x, n_y) and silently drop n_z. That's fine on flat plates
     * but must be revisited before shipping shell elements where a3 ≠ ẑ.
     */
    std::tuple<ColMatrix<T, 3>, ColMatrix<T, 3>, ColMatrix<T, 3>, Vector<T>>
    eval_local_frame(const ColMatrix<T, d>& points,
                     Index span) const requires(d == 2);

    /**
     * @brief Evaluate surface geometry at parametric points within a single span.
     *
     * Returns (a1, a2, a3, g_inv, b, jac):
     *   - a1, a2 (Q×3): raw covariant tangents a_α = ∂x/∂ξ^α.
     *   - a3     (Q×3): unit director (a1×a2)/||a1×a2||.
     *   - g_inv  (Q×3): contravariant metric components (g^11, g^12, g^22).
     *                   The covariant metric is recoverable as g_αβ = a_α·a_β.
     *   - b      (Q×3): second fundamental form (b_11, b_12, b_22),
     *                   b_αβ = a_{α,β}·a3.
     *   - jac    (Q):   √det(g) = ||a1×a2|| (surface area element).
     *
     * Pure geometry — independent of any field defined on the patch.
     */
    std::tuple<ColMatrix<T, 3>, ColMatrix<T, 3>, ColMatrix<T, 3>,
               ColMatrix<T, 3>, ColMatrix<T, 3>, Vector<T>>
    eval_surface_geometry(const ColMatrix<T, d>& points,
                          Index span) const requires(d == 2);

    /**
     * @brief Evaluate per-element surface kinematics: basis derivatives + all
     *        geometric quantities (tangents, director, metric, curvature,
     *        Christoffels) packaged in a single `SurfaceKinematics<T>` bundle.
     *
     * @param order Highest parametric-derivative order to compute (1, 2, or 3).
     *              Order 1 returns frame + jac; order 2 adds a_{αβ}, b, Γ;
     *              order 3 adds a_{αβγ}.
     *
     * Designed as the single source of truth for surface kernels: callers pull
     * what they need by name, avoiding duplicate basis evaluations and
     * duplicate Christoffel computations across `eval_covariant_derivatives`
     * and shell elements.
     */
    SurfaceKinematics<T>
    eval_surface_kinematics(const ColMatrix<T, d>& points,
                            Index span,
                            std::size_t order = 2) const requires(d == 2);

    /**
     * @brief Evaluate shape functions and their covariant derivatives w.r.t.
     *        the surface coordinates at parametric points within a single span.
     *
     * Returns std::vector<Matrix<T>> of length 3, 6, or 12 depending on order
     * (clamped to [1, 3]). Layout:
     *   [0]            N (values, Q×K)
     *   [1], [2]       N_{,1}, N_{,2}            parametric 1st = covariant 1st
     *   [3], [4], [5]  N_{|11}, N_{|12}, N_{|22}  covariant 2nd, symmetric in (α,β)
     *   [6], [7]       N_{|11,1}, N_{|11,2}       covariant 3rd, αβ = 11
     *   [8], [9]       N_{|12,1}, N_{|12,2}       covariant 3rd, αβ = 12
     *   [10], [11]     N_{|22,1}, N_{|22,2}       covariant 3rd, αβ = 22
     *
     * Covariant derivatives are computed via Christoffel correction:
     *   N_{|αβ}  = N_{,αβ}  - Γ^γ_{αβ} N_{,γ}
     *   N_{|αβγ} = ∂_γ N_{|αβ}  - Γ^σ_{γβ} N_{|ασ}  - Γ^σ_{γα} N_{|βσ}
     * with Christoffel symbols Γ^γ_{αβ} = a^γ · a_{αβ} computed internally.
     *
     * @note On curved surfaces the 3rd-order tensor is symmetric in (α,β) but
     *       NOT in (γ, α/β) — Riemann curvature breaks full symmetry. The
     *       6 entries above are the unique components.
     */
    std::vector<Matrix<T>> eval_covariant_derivatives(
        const ColMatrix<T, d>& points,
        Index span,
        std::size_t order = 2) const requires(d == 2);

    /// @brief Basis in the given parametric direction.
    const Basis<T>& basis(std::size_t dir) const 
    { return tensor_product_.basis(dir); }

    /// @brief Get a shared pointer to the 1D basis for a given parametric direction
    Ptr<const Basis<T>> basis_ptr(std::size_t dir) const 
    { return tensor_product_.basis_ptr(dir); }

    /// @brief Full tensor-product basis.
    const TensorProduct<T, d>& tensor_product() const 
    { return tensor_product_; }

    /// @brief DOF mapper for this patch.
    const DofMapper<d>& dof_mapper() const 
    { return dof_mapper_; }

    /// @brief Get the geometric dimension (always 3 for now).
    constexpr std::size_t gdim() const { return 3; }

    /// @brief Get the topological dimension.
    constexpr std::size_t tdim() const { return d; }

    /// @brief Get the control points matrix.
    const ColMatrix<T, 3>& control_pts() const { return control_pts_; }
    ColMatrix<T, 3>& control_pts() { return control_pts_; }

    /// @brief Get the number of control points.
    std::size_t num_control_pts() const { return control_pts_.rows(); }

    ColMatrix<T, 3> active_control_pts(const std::array<Index, d>& spans) const;
    ColMatrix<T, 3> get_control_points(const std::vector<Index>& indices) const;
    std::vector<Index> layer_dofs(std::size_t param_dim, bool at_start, Index layer_idx = 0) const;
    std::array<Index, d> decode_span(Index flat_idx) const;

    /// @brief Local indices of this patch's control points (0, 1, …, N−1).
    virtual std::vector<Index> global_indices() const {
        std::vector<Index> indices(num_control_pts());
        for (Index i = 0; i < indices.size(); ++i) indices[i] = i;
        return indices;
    }

    /// @brief DOF indices used for assembly scatter. Defaults to global_indices();
    ///        overridden by PatchBoundary to return the parent patch's DOF indices.
    virtual std::vector<Index> assembly_dofs() const { return global_indices(); }

protected:
    ColMatrix<T, 3> control_pts_;
    TensorProduct<T, d> tensor_product_;
    DofMapper<d> dof_mapper_;
    ColMatrix<T, d> greville_points_;
    std::vector<Index> greville_spans_;
};

template <std::floating_point T, std::size_t d>
std::vector<Matrix<T>> Patch<T, d>::eval_basis_functions(const ColMatrix<T, d>& points, Index span, std::size_t order) const
{
    auto spans = decode_span(span);
    return tensor_product_.eval_on_span(points, spans, static_cast<Index>(order));
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::active_control_pts(const std::array<Index, d>& spans) const
{
    auto dofs = dof_mapper_.get_element_dofs(spans);
    ColMatrix<T, 3> pts(dofs.size(), 3);
    for (Index i = 0; i < dofs.size(); ++i) {
        pts.row(i) = control_pts_.row(dofs[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::get_control_points(const std::vector<Index>& indices) const
{
    ColMatrix<T, 3> pts(indices.size(), 3);
    for (Index i = 0; i < indices.size(); ++i) {
        pts.row(i) = control_pts_.row(indices[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
std::vector<Index> Patch<T, d>::layer_dofs(std::size_t param_dim, bool at_start, Index layer_idx) const
{
    return dof_mapper_.get_layer_dofs(param_dim, at_start, layer_idx);
}

template <std::floating_point T, std::size_t d>
std::array<Index, d> Patch<T, d>::decode_span(Index flat_idx) const
{
    if constexpr (d == 1) {
        return {flat_idx};
    } else {
        auto intervals = tensor_product_.num_intervals();
        std::array<Index, d> spans;
        Index temp = flat_idx;
        for (std::size_t i = 0; i < d; ++i) {
            spans[i] = temp % intervals[i];
            temp /= intervals[i];
        }
        return spans;
    }
}

} // namespace pyck

#endif // PYCK_PATCH_HPP
