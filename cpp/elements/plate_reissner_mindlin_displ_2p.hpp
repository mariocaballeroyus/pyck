#ifndef PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Two-parameter rotation-free Reissner-Mindlin plate element.
 *
 * Primary variables per node:
 *   - w_b : bending contribution to the transverse displacement
 *   - ψ   : scalar Helmholtz potential for the curl part of the rotation
 *
 * Requires at least C² continuity (cubic B-splines or higher) due to the
 * third-order derivatives of w_b in the shear strain.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlinDispl2p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Material properties.
     */
    explicit PlateReissnerMindlinDispl2p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators =======================================================

    /**
     * @brief Bending strain matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Bending strain matrix.
     */
    Matrix<T> bending_strain_matrix(const Patch<T, 2>& patch,
                                    const BasisDerivs<T, 2>& basis,
                                    const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Shear strain matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Shear strain matrix.
     */
    Matrix<T> shear_strain_matrix(const Patch<T, 2>& patch,
                                  const BasisDerivs<T, 2>& basis,
                                  const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Bending constitutive matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Bending constitutive matrix.
     */
    Matrix<T> bending_constitutive_matrix(const LocalFrame<T, 2>& local,
                                          Index q) const override;

    /**
     * @brief Shear constitutive matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Shear constitutive matrix.
     */
    Matrix<T> shear_constitutive_matrix(const LocalFrame<T, 2>& local,
                                        Index q) const override;

    /**
     * @brief Displacement shape matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Displacement shape matrix.
     */
    Matrix<T> displacement_shape_matrix(const Patch<T, 2>& patch,
                                        const BasisDerivs<T, 2>& basis,
                                        const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Rotation shape matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Rotation shape matrix.
     */
    Matrix<T> rotation_shape_matrix(const Patch<T, 2>& patch,
                                    const BasisDerivs<T, 2>& basis,
                                    const LocalFrame<T, 2>& local) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (w_b + ψ).
    std::size_t num_node_dofs() const override
    { return 2; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override
    { return 3; }

    /// @brief Rotation degree of freedom indices.
    std::array<std::size_t, 2> rotation_dof_indices() const override
    { return {0, 0}; }

private:

    /// @brief Material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP
