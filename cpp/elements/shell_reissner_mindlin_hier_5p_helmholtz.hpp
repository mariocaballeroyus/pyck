#ifndef PYCK_SHELL_REISSNER_MINDLIN_HIER_5P_HELMHOLTZ_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_HIER_5P_HELMHOLTZ_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Hierarchic five-parameter rotation-free Reissner-Mindlin shell with an
 *        independent shear potential (Helmholtz decomposition of the transverse shear).
 *
 * Identical kinematics to @ref ShellReissnerMindlinHier4p, except the shear deflection
 * ``w_s`` is carried as its own nodal field rather than condensed out through
 * ``w_s = −(K_b/K_s)(Δ_g + 2νK) w_b``. The transverse shear is then the genuine
 * Helmholtz split ``γ = ∇w_s + curl ψ`` of an independent gradient potential ``w_s``
 * and curl potential ``ψ``:
 *
 *     slot 0..2 : Cartesian mid-surface displacements v_b (v_x, v_y, v_z)
 *     slot 3    : twist (curl) potential ψ
 *     slot 4    : shear (gradient) potential w_s
 *
 * Recovered mid-surface displacement u = v_b + w_s A_3. Freeing ``w_s`` drops the
 * order-3 term the 4p shear carried, so the basis need only be C¹ (degree ≥ 2).
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlinHier5pHelmholtz : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Shell material (plane-stress + transverse shear).
     */
    explicit ShellReissnerMindlinHier5pHelmholtz(Ptr<PlaneStress2d<T>> material);

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
     * @brief Displacement shape matrix (3Q × 5N). The recovered mid-surface
     *        displacement u = v_b + w_s A_3 with w_s the independent shear potential
     *        (slot 4); the twist slot is zero. Drives the distributed-load path.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Director-tilt shape matrix. Two rows per qp: the covariant tilt
     *        components w_α = −w_{b,α} + ε_α^β ψ_{,β}.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Scalar trace of the hierarchic field ψ (DOF slot 3): the nodal
     *        B-spline value placed in each node's ψ column. One row per qp.
     */
    void psi(const ElementValues<T, 2>& parent, Matrix<T>& out) const override;

    /**
     * @brief Directional derivative ∇ψ·dir = ψ_,α(A^α·dir) of the hierarchic field
     *        (DOF slot 3). With `dir` the boundary co-normal this is the normal slope
     *        ψ_,n that completes C1 continuity of ψ across a seam.
     */
    void psi_gradient(const ElementValues<T, 2>& parent,
                      const ColMatrix<T, 3>& dir, Matrix<T>& out) const override;

    /**
     * @brief Surface-normal variation δa_3·dir: the Kirchhoff–Love tilt built from the
     *        bending (Cartesian) displacement in slots 0..2; the ψ and w_s slots carry
     *        shear, not surface rotation, so they stay zero.
     */
    void director_variation(const ElementValues<T, 2>& parent,
                            const ColMatrix<T, 3>& dir, Matrix<T>& out) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (v_x, v_y, v_z, ψ, w_s).
    std::size_t num_node_dofs() const override
    { return 5; }

    /// @brief The bending covariant Hessian and the ψ curl-gradient read second
    ///        surface derivatives, so the basis is evaluated to order 2.
    Index basis_order() const override { return 2; }

    // Normal (A_3) + Curvature (B_{αβ}) + NormalDeriv1 (Weingarten A_{3,β}, for the exact
    // third fundamental form B² = A_{3,α}·A_{3,β}) + Connection (Christoffels for the
    // covariant Hessian / curl gradient). No Deriv3: w_s is independent, so the shear
    // carries no order-3 derivative of w_b.
    unsigned flags() const override
    { return Flags::Normal | Flags::Curvature | Flags::NormalDeriv1 | Flags::Connection; }

    unsigned essential_flags() const override
    { return Flags::Normal | Flags::Curvature | Flags::NormalDeriv1 | Flags::Connection; }
    unsigned natural_flags()   const override
    { return Flags::Normal | Flags::Curvature | Flags::NormalDeriv1 | Flags::Connection; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_HIER_5P_HELMHOLTZ_HPP
