#ifndef PYCK_BEAM_TIMOSHENKO_2P_HPP
#define PYCK_BEAM_TIMOSHENKO_2P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/slender_beam_1d.hpp"
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
protected:
    using idx = typename Element<T, 1>::idx;

public:

    /**
     * @brief Construct a Timoshenko beam element.
     *
     * @param material Pointer to the slender beam material model.
     */
    BeamTimoshenko2p(Ptr<SlenderBeam1d<T>> material);

    /**
     * @brief Compute the local stiffness matrix for the element.
     * @param patch The patch of the element.
     * @param q_points Quadrature points.
     * @param q_weights Quadrature weights.
     * @param span Knot-span index.
     * @param stiffness The local stiffness matrix to be computed.
     */
    void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                 const Vector<T>& jacobian,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    /// @brief Transverse displacement shape functions N_w (Q × n).
    Matrix<T> transverse_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape functions N_θ (Q × n).
    ///
    /// The rotation DOF θ shares the same isoparametric basis as w, so this
    /// returns shape_derivs[val].
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 2; }

    std::size_t min_order() const override { return 1; }

private:
    /// @brief Material and cross section geometry
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_TIMOSHENKO_2P_HPP
