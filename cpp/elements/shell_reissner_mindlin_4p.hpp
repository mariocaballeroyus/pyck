#ifndef PYCK_SHELL_REISSNER_MINDLIN_4P_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_4P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Four-parameter rotation-free Reissner-Mindlin shell element.
 *
 * The curved-surface counterpart of @ref PlateReissnerMindlinDispl2p. The
 * director tilt references the *deformed* mid-surface normal,
 * @f$\theta_\alpha=-w_{b,\alpha}-B_\alpha{}^\gamma u_\gamma
 *   +\epsilon_\alpha{}^\lambda\psi_{,\lambda}@f$ (a surface Helmholtz split plus
 * the shape-operator term that tracks the deformed normal), and the transverse
 * displacement a bending-shear split @f$u^3=w_b+w_s@f$, with the shear
 * deflection eliminated through the (leading-order) equilibrium closure
 * @f$w_s=-\tfrac{K_b}{K_s}\nabla_s^2 w_b@f$. The result is four DOFs per
 * control point — two in-plane displacement components and the two scalar
 * potentials — against five for the standard displacement-rotation shell:
 * @f[
 *   \hat{\mathbf{d}}_i = \{\,u_1,\ u_2,\ w_b,\ \psi\,\}^T .
 * @f]
 *
 * The in-plane displacement DOFs @f$(u_1,u_2)@f$ are the *covariant* surface
 * components @f$u_\alpha = \mathbf{A}_\alpha\!\cdot\mathbf{u}@f$; on the flat
 * lamina they coincide with the physical in-plane components. Like the
 * two-parameter plate the shear strain carries third derivatives of @f$w_b@f$,
 * so the basis must be @f$C^2@f$ (degree @f$\geq 3@f$).
 *
 * See `report/sections/extra.tex` for the derivation; the strain measures are
 *   @f[
 *     \varepsilon_{\alpha\beta} = \tfrac12(u_{\alpha|\beta}+u_{\beta|\alpha})
 *                                 - B_{\alpha\beta}\,\square,\quad
 *     \kappa_{\alpha\beta} = w_{b|\alpha\beta}
 *       - \tfrac12(\epsilon_\alpha{}^\delta\psi_{|\delta\beta}
 *                + \epsilon_\beta{}^\delta\psi_{|\delta\alpha})
 *       + \tfrac12(B_\alpha{}^\mu u_{\mu|\beta}+B_\beta{}^\mu u_{\mu|\alpha})
 *       + \tfrac12((B_\alpha{}^\gamma)_{|\beta}+(B_\beta{}^\gamma)_{|\alpha})u_\gamma
 *       - (B^2)_{\alpha\beta}\,\square,
 *   @f]
 *   @f[
 *     \gamma_\alpha = -\tfrac{K_b}{K_s}(\nabla_s^2 w_b)_{,\alpha}
 *                     + \epsilon_\alpha{}^\beta\psi_{,\beta},
 *   @f]
 * with @f$\square = w_b - \tfrac{K_b}{K_s}\nabla_s^2 w_b@f$ the recovered
 * transverse displacement and @f$B_{\alpha\beta}@f$ the second fundamental form.
 * Referencing the director to the *deformed* normal cancels the membrane–curvature
 * coupling @f$B_\alpha{}^\gamma u_\gamma@f$ from the shear (so @f$\gamma@f$ is
 * shear-locking-free by construction) and relocates it — consistently, together
 * with its covariant-derivative partner @f$(B_\alpha{}^\gamma)_{|\beta}@f$ — into
 * the bending strain. The @f$\nabla B@f$ term is nonzero only where curvature
 * varies (it vanishes as a tensor on flat / spherical / cylindrical surfaces) and
 * requires third derivatives of the surface (hence @ref Flags::NormalD1).
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlin4p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Shell material (plane-stress + transverse shear).
     */
    explicit ShellReissnerMindlin4p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators =======================================================

    /**
     * @brief Strain-displacement B-matrix (8Q × 4N). Eight strain rows per qp,
     *        ordered (ε_{11}, ε_{22}, 2ε_{12}, κ_{11}, κ_{22}, 2κ_{12}, γ_1, γ_2).
     */
    void strain_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Constitutive D-matrix (8×8 block-diag [D_m; D_b; D_s]).
     */
    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const override;

    // === Shape Matrices =========================================================

    /**
     * @brief Transverse-displacement shape matrix. The recovered normal
     *        displacement w = w_b − (K_b/K_s)Δ_g w_b sits in the w_b slot; the
     *        in-plane and twist slots are zero. Drives the distributed-load path.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Director-tilt shape matrix. Two rows per qp: the covariant tilt
     *        components w_α = −w_{b,α} + ε_α^β ψ_{,β}.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (u_1, u_2, w_b, ψ).
    std::size_t num_node_dofs() const override
    { return 4; }

    /// @brief Third derivatives of w_b enter the shear block, so the basis is
    ///        evaluated to order 3.
    Index basis_order() const override { return 3; }

    unsigned flags() const override
    { return Flags::Metric | Flags::Christoffels | Flags::Normal | Flags::NormalD1 | Flags::Curvature; }

    unsigned essential_flags() const override
    { return Flags::Metric | Flags::Christoffels; }
    unsigned natural_flags()   const override
    { return Flags::Metric | Flags::Christoffels | Flags::Normal | Flags::NormalD1 | Flags::Curvature; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_4P_HPP
