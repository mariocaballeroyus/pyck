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

    /**
     * @brief Strain-displacement B-matrix.
     *
     * @param ev Precomputed per-element geometric primitives.
     */
    void strain_matrix(const ElementValues<T, 2>& ev) const override;

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

    unsigned flags()           const override { return Flags::Normal | Flags::Deriv2; }
    unsigned essential_flags() const override { return Flags::None; }
    unsigned natural_flags()   const override { return Flags::Normal | Flags::Deriv2; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_KIRCHHOFF_LOVE_3P_HPP
