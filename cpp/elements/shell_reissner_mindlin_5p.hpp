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
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Strain-displacement matrix.
     */
    void strain_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Constitutive matrix.
     */
    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const override;

    // Shape Matrices =================================================================

    /**
     * @brief Displacement shape matrix stub (returns zero).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Zero matrix of correct shape.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Rotation shape matrix stub (returns zero).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Zero matrix of correct shape.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (3 displacements + 2 rotations).
    std::size_t num_node_dofs() const override
    { return 5; }

    /// @brief Minimum order of basis functions. Christoffels in the bending
    /// block require second derivatives of the surface position, i.e. an
    /// order-2 basis evaluation.
    Index basis_order() const override { return 2; }

    unsigned flags() const override { return Flags::Metric | Flags::Christoffels | Flags::Normal; }

private:

    /// @brief Shell material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_SHELL_REISSNER_MINDLIN_5P_HPP
