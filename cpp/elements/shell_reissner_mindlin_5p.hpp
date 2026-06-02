#ifndef PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
#define PYCK_SHELL_REISSNER_MINDLIN_5P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Reissner-Mindlin 5-parameter shell element on a curved surface.
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
    explicit ShellReissnerMindlin5p(Ptr<PlaneStress2d<T>> material);

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
     * @brief Rotation shape matrix.
     *
     * @param ev Per-element evaluation workspace.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (3 displacements + 2 rotations).
    std::size_t num_node_dofs() const override
    { return 5; }

    /// @brief Minimum order of basis functions. The bending block needs the
    /// second surface derivatives A_{λ,β}, i.e. an order-2 basis evaluation.
    Index basis_order() const override { return 2; }

    // Normal (A_3) + Deriv2 (order-2 surface derivatives, for the in-place normal
    // derivatives A_{3,β} and the vector-gradient kernel). Deriv1's metric + jac
    // come free via Normal. A_{3,β} is computed in place, not cached.
    unsigned flags() const override
    { return Flags::Normal | Flags::Deriv2; }

    unsigned essential_flags() const override { return Flags::None; }
    unsigned natural_flags()   const override
    { return Flags::Normal | Flags::Deriv2; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;

    /// @brief Scratch for the in-place reference-normal derivatives A_{3,β},
    ///        packed (Q·2 × 3): block β at rows β·Q.
    mutable ColMatrix<T, 3> n_d1_ws_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
