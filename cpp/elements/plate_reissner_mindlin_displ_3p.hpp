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
     * @brief Strain-displacement B-matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Strain-displacement matrix.
     */
    Matrix<T> strain_matrix(const Patch<T, 2>& patch,
                            const std::vector<Matrix<T>>& basis,
                            const IntrinsicGeometry<T, 2>& ig) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Constitutive matrix.
     */
    Matrix<T> constitutive_matrix(const IntrinsicGeometry<T, 2>& ig,
                                  Index q) const override;

    // === Shape Matrices =============================================================


    /**
     * @brief Displacement shape matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Displacement shape matrix.
     */
    Matrix<T> displacement_shape_matrix(const Patch<T, 2>& patch,
                                        const std::vector<Matrix<T>>& basis,
                                        const IntrinsicGeometry<T, 2>& ig) const override;

    /**
     * @brief Rotation shape matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Rotation shape matrix.
     */
    Matrix<T> rotation_shape_matrix(const Patch<T, 2>& patch,
                                    const std::vector<Matrix<T>>& basis,
                                    const IntrinsicGeometry<T, 2>& ig) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom.
    std::size_t num_node_dofs() const override
    { return 3; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override
    { return 2; }

private:

    /// @brief Material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP
