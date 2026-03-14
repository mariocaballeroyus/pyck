#ifndef PYCK_TIMOSHENKO_BEAM_1P_HPP
#define PYCK_TIMOSHENKO_BEAM_1P_HPP

#include <vector>

#include "element.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Single-Variable Timoshenko beam element.
/// @tparam T Scalar type.
template <std::floating_point T>
class TimoshenkoBeam1P : public Element<T, 1>
{
public:

    TimoshenkoBeam1P(T youngs_modulus,
                     T section_area,
                     T moment_inertia,
                     T shear_modulus,
                     T shear_coefficient = 5.0 / 6.0);

    /**
     * @brief Compute the local stiffness matrix for the element.
     * @param patch The patch of the element.
     * @param q_points Quadrature points.
     * @param q_weights Quadrature weights.
     * @param spans Knot-span index.
     * @param stiffness The local stiffness matrix to be computed.
     */
    void compute_local_stiffness(const Patch<T, 1>& patch,
                                 const ColMatrix<T, 1>& q_points,
                                 const Vector<T>& q_weights,
                                 Index span,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_dofs_per_node() const override { return 1; }

    std::size_t required_shape_order() const override { return 2; }

private:

    /// @brief Material Parameters
    T E_, I_, A_, G_, k_;
    T kGA_, Kb_;

};

} // namespace pyck

#endif // PYCK_TIMOSHENKO_BEAM_1P_HPP
