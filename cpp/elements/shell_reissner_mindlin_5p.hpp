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

    // Normal (A_3) + NormalDeriv1 (A_{3,β}, the cached Weingarten image −B^α_β A_α,
    // which pulls in Curvature) + Deriv2 (order-2 surface derivatives for the
    // covariant-gradient operator).
    unsigned flags() const override
    { return Flags::Normal | Flags::NormalDeriv1 | Flags::Deriv2 | Flags::Connection; }

    unsigned essential_flags() const override { return Flags::None; }
    unsigned natural_flags()   const override
    { return Flags::Normal | Flags::NormalDeriv1 | Flags::Deriv2; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
