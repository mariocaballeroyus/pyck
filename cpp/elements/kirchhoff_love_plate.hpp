#ifndef PYCK_KIRCHHOFF_LOVE_PLATE_HPP
#define PYCK_KIRCHHOFF_LOVE_PLATE_HPP

#include <vector>

#include "element.hpp"
#include "plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Kirchhoff-Love thin plate element.
///
/// Uses the tensor-product B-spline basis with C^1 continuity
/// (at minimum) to compute bending stiffness via the second
/// covariant derivatives of the basis functions.
///
/// The single DOF per control point is the transverse displacement w.
///
/// @tparam T Scalar type.
template <std::floating_point T>
class KirchhoffLovePlate : public Element<T, 2>
{
public:

    /// @brief Construct a Kirchhoff-Love plate element.
    /// @param material Pointer to the plane stress material model.
    KirchhoffLovePlate(Ptr<PlaneStress2d<T>> material);

    void compute_local_stiffness(const Patch<T, 2>& patch,
                                 const ColMatrix<T, 2>& q_points,
                                 const Vector<T>& q_weights,
                                 Index span,
                                 Matrix<T>& stiffness) const override;

    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_dofs_per_node() const override { return 1; }

    std::size_t required_shape_order() const override { return 0; }

private:
    /// @brief Material and thickness
    Ptr<PlaneStress2d<T>> material_;

    /// @brief Pre-computed 3×3 bending constitutive matrix.
    Matrix<T> D_b_;
};

} // namespace pyck

#endif // PYCK_KIRCHHOFF_LOVE_PLATE_HPP
