#ifndef PYCK_SHELL_KIRCHHOFF_LOVE_3P_HPP
#define PYCK_SHELL_KIRCHHOFF_LOVE_3P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Kirchhoff-Love thin-shell element on a curved surface (rotation-free).
 *
 * The curved-surface counterpart of `PlateKirchhoffLove1p`: transverse shear is
 * neglected, the director is the deformed surface normal, and bending is the
 * linearized change of the second fundamental form. The only unknowns are the
 * three Cartesian displacement components per control point — no rotation DOFs —
 * so the basis must be C¹ (degree ≥ 2). Membrane and bending share the
 * displacement field exactly as in `ShellReissnerMindlin5p`, but the rotation
 * DOFs and the two transverse-shear strain rows are dropped.
 *
 * Generalised strain (Voigt, 6 components):
 *     (ε₁₁, ε₂₂, 2ε₁₂, κ₁₁, κ₂₂, 2κ₁₂),  with κ_{αβ} = −δb_{αβ}.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellKirchhoffLove3p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Shell material (plane stress).
     */
    explicit ShellKirchhoffLove3p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators =======================================================

    /**
     * @brief Strain-displacement B-matrix.
     *
     * @param ev Per-element evaluation workspace.
     */
    void strain_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param ev Per-element evaluation workspace.
     * @param q Quadrature point.
     */
    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const override;

    // Shape Matrices =================================================================

    /**
     * @brief Displacement shape matrix.
     *
     * @param ev Per-element evaluation workspace.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Rotation shape matrix (not implemented — rotation-free element).
     *
     * @param ev Per-element evaluation workspace.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (3 Cartesian displacements).
    std::size_t num_node_dofs() const override
    { return 3; }

    /// @brief Minimum order of basis functions. The bending block reads the
    /// second surface derivatives A_{α,β} and the raw second basis derivatives
    /// N_{,αβ}, i.e. an order-2 basis evaluation (C¹, degree ≥ 2).
    Index basis_order() const override { return 2; }

    // Normal (A_3) + Curvature (B_{αβ}) + Deriv2 (order-2 surface derivatives A_{α,β}
    // and the raw N_{,αβ}). Curvature pulls in Deriv2/Normal/Deriv1/Values, which
    // also supply the membrane base vectors A_α and the metric the constitutive reads.
    unsigned flags() const override
    { return Flags::Normal | Flags::Curvature | Flags::Deriv2; }

    unsigned essential_flags() const override { return Flags::None; }
    unsigned natural_flags()   const override
    { return Flags::Normal | Flags::Curvature | Flags::Deriv2; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_KIRCHHOFF_LOVE_3P_HPP
