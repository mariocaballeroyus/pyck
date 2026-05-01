#ifndef PYCK_BEAM_TIMOSHENKO_1P_HPP
#define PYCK_BEAM_TIMOSHENKO_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/slender_beam_1d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Single-Variable Timoshenko beam element.
 *
 * Uses the 1-parameter formulation where rotations are eliminated
 * at the cost of C2 continuity requirements.
 *
 * DOF per node (1):
 *   - w: Transverse displacement
 *
 * Sign convention: gamma = w,x - theta = ratio * w,xxx
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class BeamTimoshenko1p : public Element<T, 1>
{
protected:
    using idx = typename Element<T, 1>::idx;

public:
    BeamTimoshenko1p(Ptr<SlenderBeam1d<T>> material);

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    T bending_stiffness() const override { return material_->bending_stiffness(); }
    T shear_stiffness() const override { return material_->shear_stiffness(); }

    /// @brief Effective transverse displacement shape functions Ñ_w (Q × n).
    ///
    /// Accounts for the Laplacian correction w = w_b - (K_b/K_s) w_b,xx, so
    /// this is: Ñ_i = N_i - (K_b/K_s) N_i,xx.
    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape matrix N_θ = -N,x (θ = -dw_b/dx).
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 1; }

    std::size_t min_order() const override { return 3; }

private:
    Ptr<SlenderBeam1d<T>> material_;

};

} // namespace pyck

#endif // PYCK_BEAM_TIMOSHENKO_1P_HPP
