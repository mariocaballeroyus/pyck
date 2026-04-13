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

    /// @brief Transverse displacement shape functions N_w (Q × n).
    Matrix<T> transverse_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape functions N_θ (Q × n).
    ///
    /// Both in-plane rotation DOFs (θ_x, θ_y) share the same isoparametric
    /// basis as the transverse displacement, so this returns shape_derivs[val].
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 3; }

    std::size_t min_order() const override { return 1; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_REISSNER_MINDLIN_PLATE_3P_HPP
