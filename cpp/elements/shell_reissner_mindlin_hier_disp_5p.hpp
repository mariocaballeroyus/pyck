#ifndef PYCK_SHELL_REISSNER_MINDLIN_HIER_DISP_5P_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_HIER_DISP_5P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Rotation-free, shear-deformable hierarchic five-parameter shell element.
 *
 * @details Oesterle, Ramm & Bischoff, "A shear deformable, rotation-free
 *          isogeometric shell formulation", CMAME 307 (2016) 235–255, §4. The
 *          Kirchhoff-Love bending displacement v_b (three Cartesian mid-surface
 *          components) is enriched by two **displacement-like** shear parameters:
 *          the total mid-surface displacement is split additively
 *          v = v_b + v^{s1} A₃ + v^{s2} A₃, with v^{sα} the through-thickness
 *          "shear displacements" along the director. Unlike the rotation-like
 *          hierarchic difference vector of Echter et al. (2013), the transverse
 *          shear strain is then a *derivative* of a displacement (2ε₁₃ = v^{s1},₁,
 *          2ε₂₃ = v^{s2},₂), one polynomial order lower than the deflection — which
 *          removes the transverse-shear-force oscillations of the rotation-based
 *          hierarchic element. Setting v^{s1} = v^{s2} = 0 recovers
 *          ShellKirchhoffLove3p exactly. (Membrane locking is unaffected — pair with
 *          a mixed/assumed-strain membrane treatment for curved shells.)
 *
 *              slot 0..2 : Cartesian bending displacements v_b (v_x, v_y, v_z)
 *              slot 3..4 : shear displacements (v^{s1}, v^{s2}) along A₃
 *
 *          **Zero-energy mode (must be suppressed).** The constant part of each shear
 *          displacement is a spurious kernel mode — redundant with the rigid transverse
 *          translation of v_b, it carries no strain (the shear is a *derivative* of
 *          v^{sα}) yet the load does work on it, so an otherwise well-posed problem is
 *          singular. As Oesterle §4.5 prescribes, anchor it by constraining the shear
 *          slots (3, 4) on (part of) the boundary — `DirectConstraint` on those slots,
 *          exactly as ShellReissnerMindlin4p's constant-ψ mode is pinned. This is
 *          required regardless of how the kinematic boundary conditions themselves are
 *          imposed (direct, penalty, Lagrange, or Nitsche); a weak (Lagrange/penalty)
 *          constraint on the *displacement* does not remove it, because the mode leaves
 *          the constrained displacement trace unchanged.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlinHierDisp5p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Shell material (plane-stress + transverse shear).
     */
    explicit ShellReissnerMindlinHierDisp5p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators =======================================================

    /// @brief Eight strain rows per qp (ε_{11}, ε_{22}, 2ε_{12}, κ_{11}, κ_{22},
    ///        2κ_{12}, γ_1, γ_2).
    Index num_strains() const override { return 8; }

    /**
     * @brief Membrane / bending / transverse-shear sub-blocks of the (8Q × 5N)
     *        strain-displacement B-matrix; stacked by the base orchestrator.
     */
    void membrane_strain_matrix(const ElementValues<T, 2>& ev, Matrix<T>& B) const override;
    void bending_strain_matrix (const ElementValues<T, 2>& ev, Matrix<T>& B) const override;
    void shear_strain_matrix   (const ElementValues<T, 2>& ev, Matrix<T>& B) const override;

    /**
     * @brief Constitutive D-matrix (8×8 block-diag [D_m; D_b; D_s]).
     */
    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const override;

    // === Shape Matrices =========================================================

    /**
     * @brief Displacement shape matrix (3Q × 5N). The mid-surface displacement is
     *        the Cartesian part v; the difference vector w is purely through-
     *        thickness and does no work against a mid-surface body force.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Director-tilt shape matrix (2Q × 5N). Covariant tilt components
     *        θ_α = (Φ×A₃)·A_α + w·A_α: the Kirchhoff-Love rotation plus the
     *        hierarchic difference vector.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Surface-normal variation δa_3·dir: the Kirchhoff–Love tilt built from the
     *        bending (Cartesian) displacement in slots 0..2; the hierarchic-displacement
     *        slots carry shear, not surface rotation, so they stay zero.
     */
    void director_variation(const ElementValues<T, 2>& parent,
                            const ColMatrix<T, 3>& dir, Matrix<T>& out) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (v_x, v_y, v_z, w^1, w^2).
    std::size_t num_node_dofs() const override
    { return 5; }

    /// @brief The Kirchhoff-Love bending of v uses the covariant Hessian (second
    ///        surface derivatives), so the basis is evaluated to order 2.
    Index basis_order() const override { return 2; }

    // Normal (A₃) + NormalDeriv1 (A_{3,β} — the third fundamental form B² and the
    // curvature B for the shear-displacement coupling; pulls in Curvature) + Deriv2
    // (order-2 surface derivatives for the covariant Hessian of v_b and the second
    // derivatives of the shear displacements) + Connection (Christoffels).
    unsigned flags() const override
    { return Flags::Normal | Flags::NormalDeriv1 | Flags::Deriv2 | Flags::Connection; }

    unsigned essential_flags() const override { return Flags::Normal; }
    unsigned natural_flags()   const override
    { return Flags::Normal | Flags::NormalDeriv1 | Flags::Deriv2 | Flags::Connection; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_HIER_DISP_5P_HPP
