#ifndef PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
#define PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Standard Kirchhoff-Love thin plate element.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateKirchhoffLove1p : public Element<T, 2>
{
public:

    // === Constructors ===========================================================

    /**
     * @brief Constructor.
     * 
     * @param material Material properties.
     */
    PlateKirchhoffLove1p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators ===========================================================

    /**
     * @brief Strain-displacement matrix B (3Q × K).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return B-matrix.
     */
    Matrix<T> strain_matrix(const Patch<T, 2>& patch,
                            const BasisDerivs<T, 2>& basis,
                            const IntrinsicGeometry<T, 2>& ig) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return D-matrix.
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
                                        const BasisDerivs<T, 2>& basis,
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
                                    const BasisDerivs<T, 2>& basis,
                                    const IntrinsicGeometry<T, 2>& ig) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (displacement + rotation).
    std::size_t num_node_dofs() const override 
    { return 1; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override
    { return 2; }

private:

    /// @brief Material properties.
    Ptr<PlaneStress2d<T>> material_;

};

} // namespace pyck

#endif // PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
