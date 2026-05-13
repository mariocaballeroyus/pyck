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

    // === Matrix Operators =======================================================

    /**
     * @brief Bending B-matrix.
     * 
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Bending B-matrix.
     */
    Matrix<T> bending_strain_matrix(const Patch<T, 2>& patch,
                                    const BasisDerivs<T, 2>& basis,
                                    const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Shear B-matrix.
     * 
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Shear B-matrix.
     */
    Matrix<T> shear_strain_matrix(const Patch<T, 2>& patch,
                                  const BasisDerivs<T, 2>& basis,
                                  const LocalFrame<T, 2>& local) const override;

    /**
     * @brief Bending constitutive D-matrix.
     * 
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Bending D-matrix.
     */
    Matrix<T> bending_constitutive_matrix(const LocalFrame<T, 2>& local,
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

    /**
     * @brief Transverse shear shape matrix.
     * 
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Transverse shear shape matrix.
     */
    Matrix<T> transverse_shear_matrix(const Patch<T, 2>& patch,
                                      const BasisDerivs<T, 2>& basis,
                                      const LocalFrame<T, 2>& local) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (displacement + rotation).
    std::size_t num_node_dofs() const override 
    { return 1; }

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

#endif // PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
