#ifndef PYCK_REISSNER_MINDLIN_PLATE_3P_HPP
#define PYCK_REISSNER_MINDLIN_PLATE_3P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Reissner-Mindlin plate element.
 *
 * Accounts for transverse shear deformation (thick plates).
 * Sign convention: kappa = grad(theta), gamma = grad(w) + theta
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlin3p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    PlateReissnerMindlin3p(Ptr<PlaneStress2d<T>> material);

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> bending_constitutive_matrix() const override { return material_->bending_matrix(); }
    Matrix<T> shear_constitutive_matrix() const override { return material_->shear_matrix(); }

    /** @brief Transverse-displacement shape matrix N_w (Q x 3n). */
    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /** @brief Rotation shape matrix N_varphi (2Q x 3n). */
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 3; }

    std::size_t min_order() const override { return 1; }

    std::array<std::size_t, 2> rotation_dof_indices() const override { return {1, 2}; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_REISSNER_MINDLIN_PLATE_3P_HPP
