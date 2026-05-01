#ifndef PYCK_BEAM_EULER_BERNOULLI_1P_HPP
#define PYCK_BEAM_EULER_BERNOULLI_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/slender_beam_1d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Euler-Bernoulli beam element.
 *
 * Pure bending formulation for thin beams.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class BeamEulerBernoulli1p : public Element<T, 1>
{
protected:
    using idx = typename Element<T, 1>::idx;

public:
    BeamEulerBernoulli1p(Ptr<SlenderBeam1d<T>> material);

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief No transverse shear strain in Euler-Bernoulli: returns a zero matrix.
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    T bending_stiffness() const override { return material_->bending_stiffness(); }

    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape matrix N_θ = -N,x (θ = -dw/dx for EB kinematics).
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 1; }

    std::size_t min_order() const override { return 2; }

private:
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_EULER_BERNOULLI_1P_HPP
