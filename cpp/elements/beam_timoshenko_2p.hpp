#ifndef PYCK_BEAM_TIMOSHENKO_2P_HPP
#define PYCK_BEAM_TIMOSHENKO_2P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/slender_beam_1d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Timoshenko beam element.
 *
 * Standard 2-parameter formulation (w, theta) accounting
 * for shear deformation.
 *
 * Sign convention: kappa = theta,x; gamma = w,x + theta
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class BeamTimoshenko2p : public Element<T, 1>
{
public:

    // === Constructors ===============================================================

    BeamTimoshenko2p(Ptr<SlenderBeam1d<T>> material);

    // === Matrix Operators ===========================================================

    /**
     * @brief Computes the bending strain matrix.
     * 
     * @param patch         The patch.
     * @param basis         The basis derivatives.
     * @param local         The local frame.
     * @return Matrix<T>    The bending strain matrix.
     */
    Matrix<T> bending_strain_matrix(const Patch<T, 1>& patch,
                                    const BasisDerivs<T, 1>& basis,
                                    const LocalFrame<T, 1>& local) const override;

    /**
     * @brief Computes the shear strain matrix.
     * 
     * @param patch         The patch.
     * @param basis         The basis derivatives.
     * @param local         The local frame.
     * @return Matrix<T>    The shear strain matrix.
     */
    Matrix<T> shear_strain_matrix(const Patch<T, 1>& patch,
                                  const BasisDerivs<T, 1>& basis,
                                  const LocalFrame<T, 1>& local) const override;

    /**
     * @brief Computes the bending constitutive matrix.
     * 
     * @param local         The local frame.
     * @param q             The quadrature point index.
     * @return T            The bending constitutive matrix.
     */
    T bending_constitutive(const LocalFrame<T, 1>& local, Index q) const override;

    /**
     * @brief Computes the shear constitutive matrix.
     * 
     * @param local         The local frame.
     * @param q             The quadrature point index.
     * @return T            The shear constitutive matrix.
     */
    T shear_constitutive(const LocalFrame<T, 1>& local, Index q) const override;

    /**
     * @brief Computes the displacement shape matrix.
     * 
     * @param patch         The patch.
     * @param basis         The basis derivatives.
     * @param local         The local frame.
     * @return Matrix<T>    The displacement shape matrix.
     */
    Matrix<T> displacement_shape_matrix(const Patch<T, 1>& patch,
                                        const BasisDerivs<T, 1>& basis,
                                        const LocalFrame<T, 1>& local) const override;

    /**
     * @brief Computes the rotation shape matrix.
     * 
     * @param patch         The patch.
     * @param basis         The basis derivatives.
     * @param local         The local frame.
     * @return Matrix<T>    The rotation shape matrix.
     */
    Matrix<T> rotation_shape_matrix(const Patch<T, 1>& patch,
                                    const BasisDerivs<T, 1>& basis,
                                    const LocalFrame<T, 1>& local) const override;

    // === Getters ====================================================================

    /// @brief Number of node degrees of freedom (displacement + rotation).
    std::size_t num_node_dofs() const override 
    { return 2; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override 
    { return 2; }

    /// @brief Index of the rotation degree of freedom.
    std::size_t rotation_dof_index() const override 
    { return 1; }

private:

    /// @brief Material properties.
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_TIMOSHENKO_2P_HPP
