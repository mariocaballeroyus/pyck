#ifndef PYCK_SHELL_REISSNER_MINDLIN_HIER_4P_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_HIER_4P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Hierarchic four-parameter rotation-free Reissner-Mindlin shell element.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlinHier4p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Shell material (plane-stress + transverse shear).
     */
    explicit ShellReissnerMindlinHier4p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators =======================================================

    /// @brief Eight strain rows per qp (ε_{11}, ε_{22}, 2ε_{12}, κ_{11}, κ_{22},
    ///        2κ_{12}, γ_1, γ_2).
    Index num_strains() const override { return 8; }

    /**
     * @brief Membrane / bending / transverse-shear sub-blocks of the (8Q × 4N)
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
     * @brief Displacement shape matrix (3Q × 4N). The recovered mid-surface
     *        displacement u = v_b + w_s A_3 with w_s = −(K_b/K_s)Δ_g w_b; the twist
     *        slot is zero. Drives the distributed-load path.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Director-tilt shape matrix. Two rows per qp: the covariant tilt
     *        components w_α = −w_{b,α} + ε_α^β ψ_{,β}.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (v_x, v_y, v_z, ψ).
    std::size_t num_node_dofs() const override
    { return 4; }

    /// @brief Third derivatives of w_b enter the shear block, so the basis is
    ///        evaluated to order 3.
    Index basis_order() const override { return 3; }

    // Normal (A_3) + Curvature (B_{αβ}) + NormalDeriv1 (Weingarten A_{3,β}, for the exact
    // third fundamental form B² = A_{3,α}·A_{3,β}) + Deriv3 (third surface derivatives for
    // the order-3 Laplace–Beltrami gradient in the shear) + Connection (Christoffels for
    // the covariant Hessian). The rotation-based bending needs no second normal derivative.
    unsigned flags() const override
    { return Flags::Normal | Flags::Curvature | Flags::NormalDeriv1 | Flags::Deriv3 | Flags::Connection; }

    unsigned essential_flags() const override
    { return Flags::Normal | Flags::Curvature | Flags::NormalDeriv1 | Flags::Deriv3 | Flags::Connection; }
    unsigned natural_flags()   const override
    { return Flags::Normal | Flags::Curvature | Flags::NormalDeriv1 | Flags::Deriv3 | Flags::Connection; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_HIER_4P_HPP