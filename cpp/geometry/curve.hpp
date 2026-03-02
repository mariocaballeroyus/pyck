#ifndef PYCK_CURVE_HPP
#define PYCK_CURVE_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "basis.hpp"
#include "patch.hpp"
#include "tensor.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief A parametric curve patch defined by a single univariate basis, embedded in 
 *        a 3-dimensional space.
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class CurvePatch : public Patch<T, 1>
{
public:

    // === Constructors & Factory Methods =============================================

    CurvePatch(Ptr<const Basis<T>> basis_u, const ColMatrix<T, 3>& control_pts)
    : Patch<T, 1>(control_pts),
      tensor_product_(std::move(basis_u)),
      dof_mapper_({tensor_product_.basis(0).num_basis()},
                  {tensor_product_.basis(0).degree()})
    {}

    // === Properties =================================================================

    /// @brief Get the basis functions for the parametric direction
    const Basis<T>& basis(std::size_t dir) const override 
    { return tensor_product_.basis(dir); }

    /// @brief Get the tensor product object
    const TensorProduct<T, 1>& tensor_product() const override 
    { return tensor_product_; }

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the raw basis functions and their parametric derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × 1) matrix.
     * @param order The highest order of derivatives to compute.
     * @return A vector of matrices where index n represents the n-th derivative d^nN/du^n.
     *         The size of each matrix is (Q, K) where K is the number of basis functions.
     * 
     *      n = 0 : [N]        (Basis functions)
     *      n = 1 : [dN/du]    (First parametric derivative)
     *      n = 2 : [d2N/du2]  (Second parametric derivative)
     *      ...
     */
    std::vector<Matrix<T>> eval_basis_functions(const ColMatrix<T, 1>& points, 
                                                std::size_t order = 0) const override;
    
    /**
     * @brief Evaluate shape functions and their physical derivatives with respect to arc-length.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × 1) matrix.
     * @param order  The highest order of physical derivatives to compute.
     * @return A vector of matrices where index n represents the n-th derivative d^nN/ds^n.
     *         The size of each matrix is (Q, K), where K is the number of basis functions.
     * 
     *      n = 0 : [N]        (Shape functions)
     *      n = 1 : [dN/ds]    (First physical derivative)
     *      n = 2 : [d2N/ds2]  (Second physical derivative)
     */          
    std::vector<Matrix<T>> eval_shape_functions(const ColMatrix<T, 1>& points,
                                                std::size_t order = 0) const override;

    /**
     * @brief Evaluate the physical curve mapping and its parametric derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × 1) matrix.
     * @param order The highest order of parametric derivatives to compute.
     * @return A vector of matrices where index n represents d^nX/du^n.
     *         The size of each matrix is (Q, 3).
     * 
     *      n = 0 : [X]        (Physical coordinates / Position)
     *      n = 1 : [dX/du]    (Parametric tangent vector)
     *      n = 2 : [d2X/du2]  (Parametric curvature vector)
     *      ...
     */
    std::vector<ColMatrix<T, 3>> eval_geometry(const ColMatrix<T, 1>& points, 
                                               std::size_t order = 0) const override;

    /**
     * @brief Evaluate the Jacobian determinant (arc-length increment) at each point.
     *
     * @param points Evaluation points in parametric coordinates as a (Q × 1) matrix.
     * @return A vector of size (Q) containing ||dX/du|| at each point.
     */
    Vector<T> eval_jacobian(const ColMatrix<T, 1>& points) const;

    // === DOF Mapping ================================================================
    
    /// @brief Get the DOF Mapper for this patch
    const DofMapper<1>& dof_mapper() const override { return dof_mapper_; }

private:

    // === Member Variables ===========================================================

    /// @brief Tensor product of the single 1D basis
    TensorProduct<T, 1> tensor_product_;

    /// @brief Geometric DOF Mapper taking local indices to global index
    DofMapper<1> dof_mapper_;

};

// === Factory Methods ================================================================

/**
 * @brief Create a straight line segment C(u) = (Lu, 0, 0).
 *        Control points are placed at Greville abscissae so that the linear
 *        mapping is represented exactly by any polynomial degree.
 *
 * @param basis B-spline basis in the u direction.
 * @param length Physical length of the line segment.
 * @return A CurvePatch representing the line segment along the x-axis.
 */
template <std::floating_point T>
CurvePatch<T> line_segment(Ptr<const Basis<T>> basis, 
                           T length);

} // namespace pyck

#endif // PYCK_CURVE_HPP
