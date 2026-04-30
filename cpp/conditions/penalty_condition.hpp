#ifndef PYCK_PENALTY_CONDITION_HPP
#define PYCK_PENALTY_CONDITION_HPP

#include <vector>
#include <Eigen/Dense>

#include "condition.hpp"
#include "boundary_patch.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Penalty method boundary condition for plate elements.
 *
 * Weakly enforces Dirichlet conditions on the physical transverse displacement
 * w and on the rotation components θ_n (normal) and θ_s (tangential) along a
 * boundary curve, using the element's own `displacement_shape_matrix` and
 * `rotation_shape_matrix`.  This makes the condition formulation-agnostic:
 *
 *   K_pen = ∫_Γ { α_w  N_w^T  N_w
 *               + α_φn N_θn^T N_θn
 *               + α_φs N_θs^T N_θs } dΓ
 *
 *   f_pen = ∫_Γ { α_w  N_w^T  w̄
 *               + α_φn N_θn^T φ̄_n
 *               + α_φs N_θs^T φ̄_s } dΓ
 *
 * For single-variable formulations (KL-1p, RM-1p) the rotation is derived
 * from the w/w_b DOF via θ = −∇w (Kirchhoff) or θ = −∇w_b (RM-1p), so
 * enabling α_φn weakly imposes a clamped-like normal-rotation constraint
 * even on 1-DOF elements.  For RM-1p the displacement row Ñ automatically
 * includes the shear-correction term −(K_b/K_s) Δw_b, so the α_w block
 * enforces the physical simply-supported constraint w = 0 (not w_b = 0).
 *
 * The outward boundary normal is computed per quadrature point from the
 * parent patch geometry; for flat plates in the xy-plane the result is exact.
 *
 * @tparam T Scalar floating-point type (double or float).
 */
template <std::floating_point T>
class PenaltyCondition : public Condition<T>
{
public:

    /**
     * @brief Construct a penalty boundary condition.
     *
     * @param boundary     1D boundary patch (extracted from a 2D surface patch
     *                     via `create_boundary`).
     * @param element      2D element formulation providing `num_node_dofs()`.
     * @param quadrature   1D Gauss–Legendre rule for boundary integration.
     * @param alpha_w      Penalty factor α_w for the transverse displacement.
     * @param w_bar        Prescribed displacement value w̄.
     * @param alpha_phi_n  Penalty factor α_φn for the normal rotation (RM-3p).
     * @param phi_n_bar    Prescribed normal rotation φ̄_n.
     * @param alpha_phi_s  Penalty factor α_φs for the tangential rotation (RM-3p).
     * @param phi_s_bar    Prescribed tangential rotation φ̄_s.
     */
    PenaltyCondition(const BoundaryPatch<T, 2>& boundary,
                     const Element<T, 2>&       element,
                     const QuadratureRule<T, 1>& quadrature,
                     T alpha_w,    T w_bar,
                     T alpha_phi_n = T(0), T phi_n_bar = T(0),
                     T alpha_phi_s = T(0), T phi_s_bar = T(0));

    /**
     * @brief Scatter the penalty stiffness and load into the global system.
     *
     * @param stiffness    Global stiffness matrix (modified in-place).
     * @param load         Global load vector (modified in-place).
     * @param layout       Equation-numbering authority.
     * @param primal_block Primal DOF block handle.
     */
    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

private:

    /// @brief Per-boundary-span local stiffness/load contribution.
    ///
    /// For formulations whose displacement shape depends on interior
    /// derivatives (e.g. RM-1p where Ñ contains ΔN), the DOFs touched by
    /// a single boundary span include all CPs active in the parent 2D
    /// element — not just those on the boundary row.  Contributions are
    /// therefore stored per-span rather than in a flat boundary-DOF block.
    /// `cps` lists the parent-patch CP indices for the span; absolute
    /// global DOFs are resolved at apply time via the layout.
    struct LocalContribution {
        Matrix<T> K;
        Vector<T> F;
        std::vector<Index> cps;
    };

    std::vector<LocalContribution> contributions_;

    /// @brief Element DOFs per control point.
    Index node_dofs_ = 0;
};

} // namespace pyck

#endif // PYCK_PENALTY_CONDITION_HPP
