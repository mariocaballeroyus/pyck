#ifndef PYCK_BEAM_TIMOSHENKO_2P_HPP
#define PYCK_BEAM_TIMOSHENKO_2P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/uniaxial_stress_1d.hpp"
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

    BeamTimoshenko2p(Ptr<UniaxialStress1d<T>> material);

    // === Matrix Operators ===========================================================

    /**
     * @brief Strain-displacement B-matrix.
     *
     * @param patch         The patch.
     * @param basis         The basis derivatives.
     * @param local         The local frame.
     * @return Matrix<T>    The strain-displacement matrix.
     */
    Matrix<T> strain_matrix(const Patch<T, 1>& patch,
                            const BasisDerivs<T, 1>& basis,
                            const LocalFrame<T, 1>& local) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param local         The local frame.
     * @param q             The quadrature point index.
     * @return Matrix<T>    Constitutive matrix.
     */
    Matrix<T> constitutive_matrix(const LocalFrame<T, 1>& local, Index q) const override;

    // === Shape Matrices =============================================================

    /**
     * @brief Displacement shape matrix.
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
     * @brief Rotation shape matrix.
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

private:

    /// @brief Material properties.
    Ptr<UniaxialStress1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_TIMOSHENKO_2P_HPP
