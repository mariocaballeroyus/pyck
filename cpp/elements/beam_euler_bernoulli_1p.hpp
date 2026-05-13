#ifndef PYCK_BEAM_EULER_BERNOULLI_1P_HPP
#define PYCK_BEAM_EULER_BERNOULLI_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/slender_beam_1d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Standard Euler-Bernoulli beam element.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class BeamEulerBernoulli1p : public Element<T, 1>
{
public:

    // === Constructors ===============================================================

    BeamEulerBernoulli1p(Ptr<SlenderBeam1d<T>> material);

    // === Matrix Operators ===========================================================

    /**
     * @brief Bending B-matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Bending B-matrix.
     */
    Matrix<T> bending_strain_matrix(const Patch<T, 1>& patch,
                                    const BasisDerivs<T, 1>& basis,
                                    const LocalFrame<T, 1>& local) const override;

    /**
     * @brief Shear B-matrix (zero for Euler-Bernoulli beams).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Shear B-matrix.
     */
    Matrix<T> shear_strain_matrix(const Patch<T, 1>& patch,
                                  const BasisDerivs<T, 1>& basis,
                                  const LocalFrame<T, 1>& local) const override;

    /**
     * @brief Bending D-matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Bending D-matrix.
     */
    T bending_constitutive(const LocalFrame<T, 1>& local, Index q) const override;

    /**
     * @brief Displacement N-matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Displacement N-matrix.
     */
    Matrix<T> displacement_shape_matrix(const Patch<T, 1>& patch,
                                        const BasisDerivs<T, 1>& basis,
                                        const LocalFrame<T, 1>& local) const override;

    /**
     * @brief Rotation N-matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @return Rotation N-matrix.
     */
    Matrix<T> rotation_shape_matrix(const Patch<T, 1>& patch,
                                    const BasisDerivs<T, 1>& basis,
                                    const LocalFrame<T, 1>& local) const override;


    // === Getters ====================================================================
    
    /// @brief Number of node degrees of freedom (displacement + rotation).
    std::size_t num_node_dofs() const override 
    { return 1; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override 
    { return 2; }

private:

    /// @brief Material properties.
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_EULER_BERNOULLI_1P_HPP
