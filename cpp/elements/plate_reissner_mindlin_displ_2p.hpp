#ifndef PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Two-parameter rotation-free Reissner-Mindlin plate element.
 *
 * Primary variables per node:
 *   - w_b : bending contribution to the transverse displacement
 *   - psi : scalar Helmholtz potential for the curl part of the rotation
 *
 * Recovered fields (with K_b = bending stiffness, K_s = shear stiffness):
 *   w     = w_b - (K_b/K_s) * Laplacian(w_b)
 *   phi_x = -w_b,x + psi,y
 *   phi_y = -w_b,y - psi,x
 *   gamma = [ -(K_b/K_s)(w_b,xxx + w_b,xyy) + psi,y,
 *             -(K_b/K_s)(w_b,xxy + w_b,yyy) - psi,x ]^T
 *   kappa = L phi
 *         = [ -w_b,xx + psi,xy,
 *             -w_b,yy - psi,xy,
 *             -2 w_b,xy + psi,yy - psi,xx ]^T
 *   m     = Db * kappa  (recovered directly from (w_b, psi); see moment_matrix)
 *
 * Requires at least C^2 continuity (cubic B-splines or higher) due to the
 * third-order derivatives of w_b in the shear strain.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlinDispl2p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    explicit PlateReissnerMindlinDispl2p(Ptr<PlaneStress2d<T>> material);

    Matrix<T> bending_constitutive_matrix() const override { return material_->bending_matrix(); }
    Matrix<T> shear_constitutive_matrix() const override { return material_->shear_matrix(); }

    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Direct moment recovery m = Db * kappa from primary DOFs (w_b, psi).
    Matrix<T> moment_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 2; }

    std::size_t min_order() const override { return 3; }

    std::array<std::size_t, 2> rotation_dof_indices() const override { return {0, 0}; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP
