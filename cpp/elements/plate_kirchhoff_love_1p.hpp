#ifndef PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
#define PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Kirchhoff-Love thin plate element.
 *
 * Uses the tensor-product B-spline basis with C^1 continuity (minimum)
 * to compute bending stiffness via the second derivatives.
 *
 * DOF per node (1):
 *   - w: Transverse displacement
 *
 * Sign convention: kappa = -grad(grad(w))
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateKirchhoffLove1p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:

    /// @brief Construct a Kirchhoff-Love plate element.
    /// @param material Pointer to the plane stress material model.
    PlateKirchhoffLove1p(Ptr<PlaneStress2d<T>> material);

    Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief No transverse shear strain in Kirchhoff-Love: returns a zero matrix.
    Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> bending_constitutive_matrix() const override { return material_->bending_matrix(); }

    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Equilibrium-based recovery of the transverse shear force from
    /// q = -div(m). For Kirchhoff-Love no shear strain block exists, so the
    /// base-class default is overridden.
    Matrix<T> transverse_shear_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 1; }

    std::size_t min_order() const override { return 2; }

    std::array<std::size_t, 2> rotation_dof_indices() const override { return {0, 0}; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
