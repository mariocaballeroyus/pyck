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
 * @brief Kirchhoff-Love thin-shell element.
 * 
 * @details Displacement-based thin-shell element with three Cartesian 
 *          displacement DOFs per node (u_x, u_y, u_z). 
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

    // === Matrix Operators ===========================================================

    /// @brief Six strain rows per qp (membrane + bending; no transverse shear).
    Index num_strains() const override { return 6; }

    /**
     * @brief Membrane / bending sub-blocks of the strain-displacement B-matrix;
     *        stacked by the base orchestrator. The shear block is the inherited
     *        no-op (Kirchhoff-Love carries no transverse shear).
     */
    void membrane_strain_matrix(const ElementValues<T, 2>& ev, Matrix<T>& B) const override;
    void bending_strain_matrix (const ElementValues<T, 2>& ev, Matrix<T>& B) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param ev Precomputed per-element geometric primitives.
     * @param q Quadrature point.
     */
    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, 
                                              Index q) const override;

    // Shape Matrices =================================================================

    /**
     * @brief Displacement shape matrix.
     *
     * @param ev Precomputed per-element geometric primitives.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Rotation shape matrix (not implemented — rotation-free element).
     *
     * @param ev Precomputed per-element geometric primitives.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    // === Properties =================================================================

    /// @brief Number of node degrees of freedom (3 Cartesian displacements).
    std::size_t num_node_dofs() const override { return 3; }

    /// @brief Minimum order of basis functions.
    Index basis_order() const override { return 2; }

    unsigned flags()           const override { return Flags::Normal | Flags::Deriv2 | Flags::Connection; }
    unsigned essential_flags() const override { return Flags::None; }
    unsigned natural_flags()   const override { return Flags::Normal | Flags::Deriv2 | Flags::Connection; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_KIRCHHOFF_LOVE_3P_HPP
