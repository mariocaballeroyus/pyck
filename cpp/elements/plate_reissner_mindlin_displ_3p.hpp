#ifndef PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Split-displacement Reissner-Mindlin plate element.
 *
 * Primary variables per node:
 *   - w_b  : bending contribution to the transverse displacement
 *   - w_s1 : shear contribution coupled to gamma_x = w_s1,x
 *   - w_s2 : shear contribution coupled to gamma_y = w_s2,y
 *
 * Recovered fields:
 *   w     = w_b + w_s1 + w_s2
 *   phi_x = -w_b,x - w_s2,x
 *   phi_y = -w_b,y - w_s1,y
 *   gamma = [w_s1,x, w_s2,y]^T
 *   kappa = L phi
 *         = [-w_b,xx - w_s2,xx,
 *            -w_b,yy - w_s1,yy,
 *            -2 w_b,xy - w_s1,xy - w_s2,xy]^T
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlinDispl3p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    explicit PlateReissnerMindlinDispl3p(Ptr<PlaneStress2d<T>> material);

    Matrix<T> bending_constitutive_matrix() const override { return material_->bending_matrix(); }
    Matrix<T> shear_constitutive_matrix() const override { return material_->shear_matrix(); }

    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 3; }

    std::size_t min_order() const override { return 2; }

    std::array<std::size_t, 2> rotation_dof_indices() const override { return {0, 0}; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP
