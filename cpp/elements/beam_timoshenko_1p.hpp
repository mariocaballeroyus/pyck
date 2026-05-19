#ifndef PYCK_BEAM_TIMOSHENKO_1P_HPP
#define PYCK_BEAM_TIMOSHENKO_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/uniaxial_stress_1d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Single-Variable Timoshenko beam element.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class BeamTimoshenko1p : public Element<T, 1>
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Constructor.
     *
     * @param material Material properties.
     */
    BeamTimoshenko1p(Ptr<UniaxialStress1d<T>> material);

    // === Matrix Operators ===========================================================

    /**
     * @brief Strain-displacement B-matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return B-matrix.
     */
    Matrix<T> strain_matrix(const Patch<T, 1>& patch,
                            const std::vector<Matrix<T>>& basis,
                            const IntrinsicGeometry<T, 1>& ig) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return D-matrix.
     */
    Matrix<T> constitutive_matrix(const IntrinsicGeometry<T, 1>& ig, Index q) const override;

    // === Shape Matrices =============================================================

    /**
     * @brief Displacement N-matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Displacement N-matrix.
     */
    Matrix<T> displacement_shape_matrix(const Patch<T, 1>& patch,
                                        const std::vector<Matrix<T>>& basis,
                                        const IntrinsicGeometry<T, 1>& ig) const override;

    /**
     * @brief Rotation N-matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Rotation N-matrix.
     */
    Matrix<T> rotation_shape_matrix(const Patch<T, 1>& patch,
                                    const std::vector<Matrix<T>>& basis,
                                    const IntrinsicGeometry<T, 1>& ig) const override;

    // === Getters ====================================================================

    /// @brief Number of node degrees of freedom (displacement + rotation).
    std::size_t num_node_dofs() const override 
    { return 1; }

    /// @brief Minimum order of basis functions.
    std::size_t min_order() const override
    { return 3; }

private:

    /// @brief Material properties.
    Ptr<UniaxialStress1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_TIMOSHENKO_1P_HPP
