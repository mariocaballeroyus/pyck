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
    /**
     * @brief Construct an Euler-Bernoulli beam element.
     *
     * @param material Pointer to the slender beam material model.
     */
    BeamEulerBernoulli1p(Ptr<SlenderBeam1d<T>> material);

    /**
     * @brief Compute the local stiffness matrix for the element.
     *
     * @param shape_fns Pre-evaluated shape functions and their derivatives.
     * @param jacobian Precomputed jacobian of the isoparametric mapping.
     * @param q_weights Quadrature weights.
     * @param stiffness The local stiffness matrix to be computed.
     */
    void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                 const Vector<T>& jacobian,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape matrix N_θ = -N,x (θ = -dw/dx for EB kinematics).
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 1; }

    std::size_t min_order() const override { return 2; }

private:
    /// @brief Material and cross section geometry
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_EULER_BERNOULLI_1P_HPP
