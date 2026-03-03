#ifndef PYCK_EULER_BERNOULLI_BEAM_1P_HPP
#define PYCK_EULER_BERNOULLI_BEAM_1P_HPP

#include <vector>

#include "element.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Euler-Bernoulli beam element with one quadrature point.
/// @tparam T Scalar type.
template <std::floating_point T>
class EulerBernoulliBeam1P : public Element<T, 1>
{
public:

    EulerBernoulliBeam1P(T youngs_modulus,
                         T section_area,
                         T moment_inertia);

    /**
     * @brief Compute the local stiffness matrix for the element.
     * @param patch The patch of the element.
     * @param q_points Quadrature points.
     * @param q_weights Quadrature weights.
     * @param spans Per-direction knot-span indices.
     * @param stiffness The local stiffness matrix to be computed (size (p+1) × (p+1)).
     */
    void compute_local_stiffness(const Patch<T, 1>& patch,
                                 const ColMatrix<T, 1>& q_points,
                                 const Vector<T>& q_weights,
                                 const std::array<Index, 1>& spans,
                                 Matrix<T>& stiffness) const override;

private:

    /// @brief Material Parameters
    T E_, I_, A_, Kb_;

};

} // namespace pyck

#endif // PYCK_EULER_BERNOULLI_BEAM_1P_HPP
