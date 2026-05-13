#ifndef PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_5P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_shell.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Reissner-Mindlin 5-parameter shell element on a curved surface.
 *
 * DOF layout per control point (5 DOFs total):
 *   slot 0..2 : Cartesian displacements (u_x, u_y, u_z)
 *   slot 3..4 : director rotation amplitudes (θ_1, θ_2) — covariant
 *               components in the local tangent basis (a_1, a_2).
 *
 * Strain measures (linearised, simplified Naghdi):
 *   ε_{αβ} = ½((∂_α u)·a_β + (∂_β u)·a_α)              membrane
 *   χ_{αβ} = ½(θ_{α|β} + θ_{β|α})                       bending
 *   γ_α    = (∂_α u)·a_3 + θ_α                          transverse shear
 *
 * The plate-style virtuals (bending_strain_matrix, displacement_shape_matrix,
 * …) are stubbed to return correctly-shaped zero matrices. The shell
 * overrides compute_local_stiffness directly to combine membrane, bending,
 * and shear blocks in a single assembly pass.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlin5p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Shell material (plane-stress + transverse shear).
     */
    explicit ShellReissnerMindlin5p(Ptr<PlaneStressShell<T>> material);

    // === Local Stiffness ========================================================

    /**
     * @brief Assemble shell K_e using covariant derivatives + surface geometry.
     *
     * Overrides the canonical base implementation to combine membrane,
     * bending, and transverse-shear contributions in a single loop.
     *
     * @param patch Patch.
     * @param elem_idx Element index.
     * @param mapped_pts Quadrature points in parametric coordinates.
     * @param q_weights Quadrature weights.
     * @param stiffness Output local stiffness matrix.
     */
    void compute_local_stiffness(const Patch<T, 2>& patch,
                                 Index elem_idx,
                                 const ColMatrix<T, 2>& mapped_pts,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    // === Matrix Operators (plate-style stubs) ===================================

    /**
     * @brief Bending strain matrix stub (returns zero).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Zero matrix of correct shape.
     */
    Matrix<T> bending_strain_matrix(const Patch<T, 2>& patch,
                                    const BasisDerivs<T, 2>& basis,
                                    const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Shear strain matrix stub (returns zero).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Zero matrix of correct shape.
     */
    Matrix<T> shear_strain_matrix(const Patch<T, 2>& patch,
                                  const BasisDerivs<T, 2>& basis,
                                  const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Bending constitutive matrix stub (returns zero).
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Zero matrix.
     */
    Matrix<T> bending_constitutive_matrix(const LocalFrame<T, 2>& local,
                                          Index q) const override;

    /**
     * @brief Displacement shape matrix stub (returns zero).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Zero matrix of correct shape.
     */
    Matrix<T> displacement_shape_matrix(const Patch<T, 2>& patch,
                                        const BasisDerivs<T, 2>& basis,
                                        const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Rotation shape matrix stub (returns zero).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Zero matrix of correct shape.
     */
    Matrix<T> rotation_shape_matrix(const Patch<T, 2>& patch,
                                    const BasisDerivs<T, 2>& basis,
                                    const LocalFrame<T, 2>& local) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (3 displacements + 2 rotations).
    std::size_t num_node_dofs() const override
    { return 5; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override
    { return 1; }

    /// @brief Index of the first Cartesian displacement DOF (u_x).
    std::size_t displacement_dof_index() const override
    { return 0; }

    /// @brief Rotation degree of freedom indices (θ_1, θ_2).
    std::array<std::size_t, 2> rotation_dof_indices() const override
    { return {3, 4}; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStressShell<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
