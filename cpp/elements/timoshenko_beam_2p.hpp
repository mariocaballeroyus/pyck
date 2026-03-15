#ifndef PYCK_TIMOSHENKO_BEAM_2P_HPP
#define PYCK_TIMOSHENKO_BEAM_2P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/slender_beam_1d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Timoshenko beam element with two variables (w, theta).
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class TimoshenkoBeam2p : public Element<T, 1>
{
public:
    using idx = typename Element<T, 1>::idx;

    /**
     * @brief Construct a Timoshenko beam element.
     *
     * @param material Pointer to the beam material model.
     */
    TimoshenkoBeam2p(Ptr<SlenderBeam1d<T>> material);

    /**
     * @brief Compute the local stiffness matrix for the element.
     * @param patch The patch of the element.
     * @param q_points Quadrature points.
     * @param q_weights Quadrature weights.
     * @param span Knot-span index.
     * @param stiffness The local stiffness matrix to be computed.
     */
    void compute_local_stiffness(const Patch<T, 1>& patch,
                                 const ColMatrix<T, 1>& q_points,
                                 const Vector<T>& q_weights,
                                 Index span,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_dofs_per_node() const override { return 2; }

private:
    /// @brief Material and cross section geometry
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_TIMOSHENKO_BEAM_2P_HPP
