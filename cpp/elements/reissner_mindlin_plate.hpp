#ifndef PYCK_REISSNER_MINDLIN_PLATE_HPP
#define PYCK_REISSNER_MINDLIN_PLATE_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Reissner-Mindlin plate element (3 DOFs per node: w, theta_x, theta_y).
 *
 * This element accounts for transverse shear deformation.
 * w: transverse displacement
 * theta_x: rotation about y-axis
 * theta_y: rotation about x-axis (with sign convention such that gamma = grad(w) - theta)
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ReissnerMindlinPlate : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    /**
     * @brief Construct a Reissner-Mindlin plate element.
     *
     * @param material Pointer to the plane stress material model.
     * @param k_s Shear correction factor (default 5/6).
     */
    ReissnerMindlinPlate(Ptr<PlaneStress2d<T>> material, T k_s = T(5.0 / 6.0));

    void compute_local_stiffness(const Patch<T, 2>& patch,
                                 const ColMatrix<T, 2>& q_points,
                                 const Vector<T>& q_weights,
                                 Index span,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_dofs_per_node() const override { return 3; }

    std::size_t required_shape_order() const override { return 1; }

private:
    Ptr<PlaneStress2d<T>> material_;
    T k_s_;
    Matrix<T> D_b_; ///< Bending constitutive matrix (3x3)
    Matrix<T> D_s_; ///< Shear constitutive matrix (2x2)
};

} // namespace pyck

#endif // PYCK_REISSNER_MINDLIN_PLATE_HPP
