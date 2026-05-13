#ifndef PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Split-displacement Reissner-Mindlin plate element.
 *
 * Primary variables per node:
 *   - w_b  : bending contribution to the transverse displacement
 *   - w_s1 : shear contribution carrying γ_1 = w_{s1,1}
 *   - w_s2 : shear contribution carrying γ_2 = w_{s2,2}
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlinDispl3p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     *
     * @param material Material properties.
     */
    explicit PlateReissnerMindlinDispl3p(Ptr<PlaneStress2d<T>> material);

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

    /// @brief Number of node degrees of freedom (w_b + w_s1 + w_s2).
    std::size_t num_node_dofs() const override
    { return 3; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override
    { return 2; }

    /// @brief Rotation degree of freedom indices.
    std::array<std::size_t, 2> rotation_dof_indices() const override
    { return {0, 0}; }

private:

    /// @brief Material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP
