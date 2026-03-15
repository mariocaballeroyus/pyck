#ifndef PYCK_EULER_BERNOULLI_BEAM_1P_HPP
#define PYCK_EULER_BERNOULLI_BEAM_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/slender_beam_1d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Euler-Bernoulli beam element with one quadrature point.
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class EulerBernoulliBeam1p : public Element<T, 1>
{
public:
    using idx = typename Element<T, 1>::idx;
    /**
     * @brief Construct an Euler-Bernoulli beam element.
     *
     * @param material Pointer to the beam material model.
     */
    EulerBernoulliBeam1p(Ptr<SlenderBeam1d<T>> material);

    /**
     * @brief Compute the local stiffness matrix for the element.
     * @param patch The patch of the element.
     * @param q_points Quadrature points.
     * @param q_weights Quadrature weights.
     * @param spans Knot-span index.
     * @param stiffness The local stiffness matrix to be computed (size (p+1) × (p+1)).
     */
    void compute_local_stiffness(const Patch<T, 1>& patch,
                                 const ColMatrix<T, 1>& q_points,
                                 const Vector<T>& q_weights,
                                 Index span,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_dofs_per_node() const override { return 1; }

private:
    /// @brief Material and cross section geometry
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_EULER_BERNOULLI_BEAM_1P_HPP
