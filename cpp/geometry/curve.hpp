#ifndef PYCK_CURVE_HPP
#define PYCK_CURVE_HPP

#include <cstddef>
#include <vector>
#include <Eigen/Dense>

#include "bspline.hpp"
#include "patch.hpp"
#include "tensor.hpp"

namespace pyck
{

/**
 * @brief A parametric curve patch defined by a single univariate basis, embedded in 
 *        a gdim-dimensional space.
 */
class CurvePatch : public Patch
{
public:

    // === Constructors & Factory Methods =============================================

    CurvePatch(std::size_t gdim,
           std::shared_ptr<Basis> basis_u,
           const Eigen::MatrixXd& control_pts)
    : Patch(gdim, 1, control_pts),
      tensor_product_({std::move(basis_u)})
    {}

    /// @brief Non-owning constructor from a raw Basis pointer.
    CurvePatch(std::size_t gdim,
           Basis* basis_u,
           const Eigen::MatrixXd& control_pts)
    : Patch(gdim, 1, control_pts),
      tensor_product_({std::shared_ptr<Basis>(std::shared_ptr<Basis>{}, basis_u)})
    {}

    /**
     * @brief Create a straight line segment C(u) = (Lu, 0, 0).
     *        Control points are placed at Greville abscissae so that the linear
     *        mapping is represented exactly by any polynomial degree.
     *
     * @param basis B-spline basis in the u direction.
     * @param length Physical length of the line segment.
     * @param gdim Geometric dimension (default = 3).
     * @return A CurvePatch representing the line segment along the x-axis.
     */
    static CurvePatch line_segment(BSpline* basis, double length, size_t gdim = 3);

    // === Properties =================================================================

    /**
     * @brief Get the basis functions for a given parametric direction.
     * @param dir The parametric direction (must be 0 for curves).
     */
    const Basis& basis(std::size_t dir) const override { return tensor_product_.basis(dir); }

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
    std::vector<Eigen::MatrixXd> eval_basis_functions(const Eigen::MatrixXd& points, 
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
    std::vector<Eigen::MatrixXd> eval_shape_functions(const Eigen::MatrixXd& points,
                                                      std::size_t order = 0) const override;

    // No eval_physical_derivs or unordered_map. Only Patch interface functions implemented.

    /**
     * @brief Evaluate the physical curve mapping and its parametric derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × 1) matrix.
     * @param order The highest order of parametric derivatives to compute.
     * @return A vector of matrices where index n represents d^nX/du^n.
     *         The size of each matrix is (Q, gdim).
     * 
     *      n = 0 : [X]        (Physical coordinates / Position)
     *      n = 1 : [dX/du]    (Parametric tangent vector)
     *      n = 2 : [d2X/du2]  (Parametric curvature vector)
     *      ...
     */
    std::vector<Eigen::MatrixXd> eval_geometry(const Eigen::MatrixXd& points, 
                                               std::size_t order = 0) const override;

    /**
     * @brief Evaluate the Jacobian determinant (arc-length increment) at each point.
     *
     * @param points Evaluation points in parametric coordinates as a (Q × 1) matrix.
     * @return A vector of size (Q) containing ||dX/du|| at each point.
     */
    Eigen::VectorXd eval_jacobian(const Eigen::MatrixXd& points) const override;

private:
    /// @brief Tensor product of the single 1D basis
    TensorProduct tensor_product_;

};

} // namespace pyck

#endif // PYCK_CURVE_HPP
