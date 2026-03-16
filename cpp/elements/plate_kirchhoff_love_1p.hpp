#ifndef PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
#define PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP

#include <vector>

#include "element.hpp"
#include "plane_stress_2d.hpp"
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
 * Sign convention: kappa = grad(grad(w))
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

    void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                 const Vector<T>& jacobian,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 1; }

    std::size_t min_order() const override { return 2; }

private:
    /// @brief Material and thickness
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
