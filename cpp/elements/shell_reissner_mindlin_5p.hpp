#ifndef PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_5P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_shell.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Reissner–Mindlin 5-parameter shell element on a curved surface.
 *
 * DOF layout per control point (5 DOFs total):
 *   slot 0..2 : Cartesian displacements (u_x, u_y, u_z)
 *   slot 3..4 : director rotation amplitudes (θ_1, θ_2) — covariant
 *               components in the local tangent basis (a_1, a_2).
 *
 * Strain measures (linearised, simplified Naghdi):
 *   ε_{αβ} = ½((∂_α u)·a_β + (∂_β u)·a_α)              membrane
 *   χ_{αβ} = ½(∂_α θ_β + ∂_β θ_α) − Γ^γ_{αβ} θ_γ        bending
 *   γ_α    = (∂_α u)·a_3 + θ_α                         transverse shear
 *
 * Constitutive law: isotropic plane-stress plus transverse shear, exposed
 * through `PlaneStressShell` per quadrature point.
 *
 * Locking: full Gauss integration; membrane/shear locking is a known
 * limitation in this baseline formulation. See docs/shell_reissner_mindlin_5p.md.
 *
 * @note The plate-style virtuals (`bending_strain_matrix`, `displacement_shape_matrix`,
 *       …) are stubbed to return correctly-shaped zero matrices. They are kept
 *       only so existing flat-plate boundary conditions don't crash if applied
 *       to a shell; shell-aware boundary conditions are a follow-up.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlin5p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    ShellReissnerMindlin5p(Ptr<PlaneStressShell<T>> material);

    /// @brief Override: assemble shell K_e using covariant derivatives + surface geometry.
    void compute_local_stiffness(const Patch<T, 2>& patch,
                                 Index elem_idx,
                                 const ColMatrix<T, 2>& mapped_pts,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    std::size_t num_node_dofs() const override { return 5; }
    std::size_t min_order()     const override { return 1; }
    std::size_t displacement_dof_index() const override { return 0; }   // u_x
    std::array<std::size_t, 2> rotation_dof_indices() const override { return {3, 4}; }

    // ----- Plate-style virtuals: stubs returning zero matrices of the right shape.
    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> shear_strain_matrix  (const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> bending_constitutive_matrix() const override;
    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> rotation_shape_matrix    (const std::vector<Matrix<T>>& shape_derivs) const override;

private:
    Ptr<PlaneStressShell<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
