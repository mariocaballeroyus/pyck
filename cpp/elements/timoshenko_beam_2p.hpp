#ifndef PYCK_TIMOSHENKO_BEAM_2P_HPP
#define PYCK_TIMOSHENKO_BEAM_2P_HPP

#include <vector>

#include "element.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Timoshenko beam element with one quadrature point handling shear and bending.
/// @tparam T Scalar type.
template <std::floating_point T>
class TimoshenkoBeam2P : public Element<T, 1>
{
    using idx = typename Element<T, 1>::idx;

public:
    /**
     * @brief Construct a Timoshenko beam element.
     *
     * @param youngs_modulus Young's modulus E.
     * @param section_area Cross-section area A.
     * @param moment_inertia Moment of inertia I.
     * @param shear_modulus Shear modulus G.
     * @param shear_coefficient Shear coefficient k.
     */
    TimoshenkoBeam2P(T youngs_modulus,
                     T section_area,
                     T moment_inertia,
                     T shear_modulus,
                     T shear_coefficient = 5.0 / 6.0);

    void compute_local_stiffness(const Patch<T, 1>& patch,
                                 const ColMatrix<T, 1>& q_points,
                                 const Vector<T>& q_weights,
                                 Index span,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_dofs_per_node() const override { return 2; }

private:
    /// @brief Material Parameters
    T E_, I_, A_, G_, k_, Ks_, Kb_;

};

} // namespace pyck

#endif // PYCK_TIMOSHENKO_BEAM_2P_HPP
