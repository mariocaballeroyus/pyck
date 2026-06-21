#ifndef PYCK_SHELL_REISSNER_MINDLIN_HIER_5P_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_HIER_5P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Hierarchic five-parameter Reissner-Mindlin shell element.
 *
 * @details Echter, Oesterle & Bischoff (2013), "A hierarchic family of
 *          isogeometric shell finite elements", §2.2. The Kirchhoff-Love
 *          displacement field (three Cartesian mid-surface displacements) is
 *          enriched by a hierarchic difference vector w = w^λ A_λ added to the
 *          *rotated* director a₃ = (A₃ + Φ×A₃) + w. The split into bending and
 *          shear deformations makes the element free of transverse-shear locking
 *          by construction: setting w = 0 recovers ShellKirchhoffLove3p exactly.
 *
 *              slot 0..2 : Cartesian displacements (v_x, v_y, v_z)
 *              slot 3..4 : hierarchic difference vector (w^1, w^2)
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlinHier5p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Shell material (plane-stress + transverse shear).
     */
    explicit ShellReissnerMindlinHier5p(Ptr<PlaneStress2d<T>> material);

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

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (v_x, v_y, v_z, w^1, w^2).
    std::size_t num_node_dofs() const override
    { return 5; }

    /// @brief The Kirchhoff-Love bending of v uses the covariant Hessian (second
    ///        surface derivatives), so the basis is evaluated to order 2.
    Index basis_order() const override { return 2; }

    // Normal (A₃) + Deriv2 (order-2 surface derivatives for the covariant Hessian
    // of v and the covariant gradient of w) + Connection (Christoffels for both).
    unsigned flags() const override
    { return Flags::Normal | Flags::Deriv2 | Flags::Connection; }

    unsigned essential_flags() const override { return Flags::Normal; }
    unsigned natural_flags()   const override
    { return Flags::Normal | Flags::Deriv2 | Flags::Connection; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_HIER_5P_HPP
