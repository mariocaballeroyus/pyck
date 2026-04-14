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
    /**
     * @brief Construct a Reissner-Mindlin plate element.
     *
     * @param material Pointer to the plane stress material model.
     */
    PlateReissnerMindlin3p(Ptr<PlaneStress2d<T>> material);

    void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                 const Vector<T>& jacobian,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    /// @brief Transverse-displacement shape matrix N_w (Q × n).
    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape matrix (2Q × n): rows `2q`, `2q+1` are both the
    /// isoparametric basis (same for N_θx and N_θy).  θ_x and θ_y live on
    /// their own DOF slots (1, 2) — see `rotation_dof_indices`.
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 3; }

    std::size_t min_order() const override { return 1; }

    std::array<std::size_t, 2> rotation_dof_indices() const override { return {1, 2}; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_REISSNER_MINDLIN_PLATE_3P_HPP
