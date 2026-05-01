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

    BeamTimoshenko2p(Ptr<SlenderBeam1d<T>> material);

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    T bending_stiffness() const override { return material_->bending_stiffness(); }
    T shear_stiffness() const override { return material_->shear_stiffness(); }

    /// @brief Transverse-displacement shape matrix N_w (Q × n).
    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape matrix N_θ (Q × n).
    ///
    /// The rotation DOF θ shares the same isoparametric basis as w; it lives
    /// on DOF slot 1 (see `rotation_dof_index`).
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 2; }

    std::size_t min_order() const override { return 1; }

    std::size_t rotation_dof_index() const override { return 1; }

private:
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_TIMOSHENKO_2P_HPP
