#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <tuple>
#include <type_traits>
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

// === Basis Derivatives ==============================================================

/**
 * @brief Parametric basis derivatives at a set of quadrature points 
 * defined on a single span of a patch.
 */
template <std::floating_point T, std::size_t d>
struct BasisDerivs;

/**
 * @brief Parametric basis derivatives defined on 1-dimensional patches
 */
template <std::floating_point T>
struct BasisDerivs<T, 1>
{
    Matrix<T> N;          ///< values                                             (Q×K)
    Matrix<T> N_u;        ///< 1st derivative N_{,u}             (order ≥ 1)      (Q×K)
    Matrix<T> N_uu;       ///< 2nd derivative N_{,uu}            (order ≥ 2)      (Q×K)
    Matrix<T> N_uuu;      ///< 3rd derivative N_{,uuu}           (order ≥ 3)      (Q×K)
    Index     order = 0;
};

/**
 * Basis basis derivatives defined on 2-dimensional patches
 */
template <std::floating_point T>
struct BasisDerivs<T, 2>
{
    Matrix<T> N;                           ///< values                            (Q×K)
    Matrix<T> N_u,   N_v;                  ///< 1st derivatives  (order ≥ 1)      (Q×K)
    Matrix<T> N_uu,  N_uv,  N_vv;          ///< 2nd derivatives  (order ≥ 2)      (Q×K)
    Matrix<T> N_uuu, N_uuv, N_uvv, N_vvv;  ///< 3rd derivatives  (order ≥ 3)      (Q×K)
    Index     order = 0;
};

// === Local Frames ===================================================================

/**
 * @brief Per-quadrature-point local geometric frame.
 */
template <std::floating_point T, std::size_t d>
struct LocalFrame;

/**
 * @brief 1D local geometric frame.
 */
template <std::floating_point T>
struct LocalFrame<T, 1>
{
    ColMatrix<T, 3> a1;              ///< tangent a_1                             (Q×3)
    ColMatrix<T, 3> a11;             ///< tangent derivative a_{1,1}  (order ≥ 2) (Q×3)
    ColMatrix<T, 3> a111;            ///< 2nd tangent deriv a_{1,11}  (order ≥ 3) (Q×3)
    Vector<T>       g11;             ///< metric g_{11}                           (Q)
    Vector<T>       g_inv_11;        ///< inverse metric g^{11}                   (Q)
    Vector<T>       jac;             ///< \sqrt{g_{11}}                           (Q)
};

/**
 * @brief 2D local geometric frame.
 */
template <std::floating_point T>
struct LocalFrame<T, 2>
{
    ColMatrix<T, 3> a1, a2;                  ///< covariant tangents a_α          (Q×3)
    ColMatrix<T, 3> a11, a12, a22;           ///< 1st derivs a_{αβ}    (order ≥ 2) (Q×3)
    ColMatrix<T, 3> a111, a112, a122, a222;  ///< 2nd derivs a_{αβγ}   (order ≥ 3) (Q×3)
    ColMatrix<T, 3> g;                       ///< metric (g_11, g_12, g_22)       (Q×3)
    ColMatrix<T, 3> g_inv;                   ///< inverse metric (g^11, g^12, g^22)(Q×3)
    Vector<T>       jac;                     ///< √det g                          (Q)
};

// === Christoffel Symbols ============================================================

/**
 * @brief Christoffel symbols Γ^γ_{αβ} at each quadrature point on a single
 *        span of a patch.
 *
 * Built by `Patch::eval_christoffel(local)` from a precomputed LocalFrame.
 * Specialised per parametric dimension: 1D has a single scalar Γ¹_{11};
 * 2D bundles the six independent components (Γ is symmetric in αβ).
 */
template <std::floating_point T, std::size_t d>
struct ChristoffelSymbols;

/**
 * @brief 1D Christoffel symbol Γ^1_{11} = (a_1 · a_{1,1}) / g_{11}.
 */
template <std::floating_point T>
struct ChristoffelSymbols<T, 1>
{
    Vector<T> G1_11;                 ///< Γ^1_{11}                                 (Q)
};

/**
 * @brief 2D Christoffel symbols Γ^γ_{αβ} = a^γ · a_{αβ}, symmetric in αβ.
 *        Six independent components per quadrature point.
 */
template <std::floating_point T>
struct ChristoffelSymbols<T, 2>
{
    Vector<T> G1_11, G1_12, G1_22;   ///< Γ^1_{αβ}                                 (Q)
    Vector<T> G2_11, G2_12, G2_22;   ///< Γ^2_{αβ}                                 (Q)
};

// === Laplace-Beltrami gradient partials (2D) =======================================

/**
 * @brief Per-quadrature-point auxiliary data for ∂_α(Δ_g f) of a 2D scalar f.
 */
template <std::floating_point T>
struct LaplaceGradAux
{
    Vector<T> G11_d1, G12_d1, G22_d1;   ///< (g^{βγ})_{,1}                        (Q)
    Vector<T> G11_d2, G12_d2, G22_d2;   ///< (g^{βγ})_{,2}                        (Q)
    Vector<T> c1, c2;                   ///< c^δ ≡ g^{βγ} Γ^δ_{βγ}                (Q)
    Vector<T> c1_d1, c2_d1;             ///< (c^δ)_{,1}                          (Q)
    Vector<T> c1_d2, c2_d2;             ///< (c^δ)_{,2}                          (Q)
};

/**
 * @brief Build a LaplaceGradAux<T> at all quadrature points from a precomputed
 *        2D LocalFrame and ChristoffelSymbols. Requires `local.a111…a222` (the
 *        basis must have been evaluated at order ≥ 3).
 */
template <std::floating_point T>
LaplaceGradAux<T> compute_laplace_grad_aux(const LocalFrame<T, 2>& local,
                                            const ChristoffelSymbols<T, 2>& chr);

// === Patches ========================================================================

/**
 * @brief Primary template for parametric patches defined by the tensor-product
 *        of univariate basis functions.
 * 
 * @details This class provides common data and logic for all patches (curves, surfaces, etc.).
 * Mathematical operations like eval_shape_functions are specialized for each dimension.
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

    /**
     * @brief Virtual destructor.
     */
    virtual ~Patch() = default;

    // ==== Basis Evaluation ==========================================================

    /**
     * @brief Evaluate shape functions and their derivatives at the given coordinates 
     *        in the parametric domain.
     * 
     * @param eval_coords Coordinates in the parametric domain.
     * @param span_idx Index of the span to evaluate.
     * @param derivs_order Highest order of derivatives to compute.
     * @return A struct containing the basis functions and their derivatives.
     */
    BasisDerivs<T, d> 
    eval_basis(const ColMatrix<T, d>& eval_coords,
               Index span_idx,
               std::size_t derivs_order = 0) const requires(d == 1);
    BasisDerivs<T, d> 
    eval_basis(const ColMatrix<T, d>& eval_coords,
               Index span_idx,
               std::size_t derivs_order = 0) const requires(d == 2);

    // === Local Frame Evaluation =====================================================

    /**
     * @brief Evaluate the local geometric frame at the given coordinates in the 
     *        parametric domain.
     * 
     * @param basis The basis functions and their derivatives.
     * @param act_pts The active control points.
     * @return A struct containing the local geometric frame.
     */
    LocalFrame<T, d> 
    eval_local_frame(const BasisDerivs<T, d>& basis,
                     const ColMatrix<T, 3>& act_pts) const requires(d == 1);
    LocalFrame<T, d> 
    eval_local_frame(const BasisDerivs<T, d>& basis,
                     const ColMatrix<T, 3>& act_pts) const requires(d == 2);

    
    // === Christoffel Symbols Evaluation =============================================

    /**
     * @brief Christoffel symbols Γ^γ_{αβ} at each quadrature point.
     *
     * 1D returns the single non-zero symbol Γ¹_{11}; 2D returns all six
     * independent components Γ^γ_{αβ} (symmetric in αβ).
     */
    ChristoffelSymbols<T, d> 
    eval_christoffel(const LocalFrame<T, d>& local) const requires(d == 1);
    ChristoffelSymbols<T, d>
    eval_christoffel(const LocalFrame<T, d>& local) const requires(d == 2);

    // === Normal Evaluation ========================================================

    /**
     * @brief Director vectors for a parametric surface via cross product of the 
     *        tangent vectors.
     */ 
    ColMatrix<T, 3> 
    eval_normal(const LocalFrame<T, 2>& local) const requires(d == 2);

    /** 
     * @brief Director derivatives (a_{3,1}, a_{3,2}) via Weingarten:
     *        a_{3,β} = -g^{γδ} (a_{δβ}·a_3) a_γ.
     */
    std::pair<ColMatrix<T, 3>, ColMatrix<T, 3>>
    eval_normal_deriv(const LocalFrame<T, 2>& local,
                      const ColMatrix<T, 3>& a_3) const requires(d == 2);

    // === Getter Functions ===========================================================

    /// @brief Get the Greville points of the patch in the parametric domain.
    const ColMatrix<T, d>& greville_points() const 
    { return greville_points_; }

    /// @brief Get the element span containing the Greville point for a global DOF index.
    Index greville_span(Index dof_index) const 
    { return greville_spans_[dof_index]; }

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

    /// @brief Get the control points matrix (const).
    const ColMatrix<T, 3>& control_pts() const 
    { return control_pts_; }

    /// @brief Get the control points matrix (non-const).
    ColMatrix<T, 3>& control_pts() 
    { return control_pts_; }

    /// @brief Get the number of control points.
    std::size_t num_control_pts() const 
    { return control_pts_.rows(); }

    ColMatrix<T, 3> active_control_pts(const std::array<Index, d>& spans) const;

    /// @brief Convenience overload: flat span index. Decodes internally.
    ColMatrix<T, 3> active_control_pts(Index span_idx) const
    { return this->active_control_pts(this->decode_span(span_idx)); }

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
