#ifndef PYCK_PLATE_REISSNER_MINDLIN_1P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Single-variable Reissner-Mindlin plate element.
 *
 * Uses a bending potential w_b as the sole primary variable.
 * The total deflection is recovered via a Laplacian correction:
 *   w = w_b - (K_b / K_s) * Laplacian(w_b)
 *
 * Requires at least C^2 continuity (cubic B-splines or higher)
 * to satisfy the third-order derivative requirements.
 *
 * DOF per node (1):
 *   - w_b: Bending potential
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlin1p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:

    PlateReissnerMindlin1p(Ptr<PlaneStress2d<T>> material);

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> bending_constitutive_matrix() const override { return material_->bending_matrix(); }
    Matrix<T> shear_constitutive_matrix() const override { return material_->shear_matrix(); }

    /** @brief Effective transverse-displacement shape matrix N_w_tilde (Q x n), including the Laplacian shear correction. */
    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /** @brief Rotation shape matrix N_varphi (2Q x n). */
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 1; }

    std::size_t min_order() const override { return 3; }

    std::array<std::size_t, 2> rotation_dof_indices() const override { return {0, 0}; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_1P_HPP
