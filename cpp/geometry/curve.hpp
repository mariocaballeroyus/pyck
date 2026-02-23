#ifndef PYCK_CURVE_HPP
#define PYCK_CURVE_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

#include "bspline.hpp"
#include "patch.hpp"
#include "tensor.hpp"

namespace pyck
{

class CurvePatch : public Patch
{
public:

    // --- Constructors & Factory Methods ------------------------------------------------

    /**
     * @brief Construct a curve patch.
     * 
     * @param gdim  Geometric dimension (e.g. 2 for a planar curve, 3 for a space curve).
     * @param basis Basis functions in the single parametric direction u.
     * @param control_points  Control-point matrix of shape n × gdim.
     */
    CurvePatch(std::size_t gdim,
               Basis* basis,
               const Eigen::MatrixXd& control_points)
        : Patch(gdim, 1, control_points),
          tensor_product_({
              std::shared_ptr<Basis>(std::shared_ptr<Basis>{}, basis)
          }) {}

    /**
     * @brief Create a straight line segment C(u) = (L·u, 0, 0).
     *
     * Control points are placed at Greville abscissae so that the linear
     * mapping is represented exactly by any polynomial degree.
     *
     * @param basis B-spline basis in the u direction.
     * @param L     Physical length of the line segment.
     * @return A 3D CurvePatch representing the line segment along the x-axis.
     */
    static CurvePatch line_segment(BSpline* basis, double L);

    // --- Properties ----------------------------------------------------------------

    /// @brief Get the basis in a given parametric direction (only dir=0 valid for curves).
    const Basis& basis(std::size_t dir) const override { return tensor_product_.basis(dir); }

    // --- Evaluation ----------------------------------------------------------------

    /**
     * @brief Evaluate the curve at given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 1). Each row is a u value.
     * @return Position matrix of shape (Q × gdim), where row i is C(u_i).
     */
    Eigen::MatrixXd eval(const Eigen::MatrixXd& params) const;

    /**
     * @brief Evaluate physical (arc-length) derivatives of basis functions.
     *
     * Builds a local tangent frame at each evaluation point via the tangent
     * vector dC/du. Derivatives are w.r.t. arc-length coordinate x along
     * the curve — NOT the parametric coordinate u.
     *
     * @param params  Evaluation points, shape (Q × 1). Each row is a u value.
     * @param order   Maximum derivative order (1, 2, or 3). Default = 1.
     * @return An unordered map from derivative label to a (Q × n_basis) matrix.
     *
     *         Order 1 keys: "N", "dNdx"
     *         Order 2 adds: "dNdxdx"
     *         Order 3 adds: "dNdxdxdx"
     */
    std::unordered_map<std::string, Eigen::MatrixXd>
    eval_physical_derivs(const Eigen::MatrixXd& params,
                         std::size_t order = 1) const;

    /**
     * @brief Compute the Jacobian (tangent vector) at the given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 1). Each row is a u value.
     * @return A vector of Q matrices, each of shape (gdim × 1).
     *         The single column is dC/du.
     */
    std::vector<Eigen::MatrixXd> jacobian(const Eigen::MatrixXd& params) const override;

    /**
     * @brief Compute the Jacobian determinant at the given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 1). Each row is a u value.
     * @return VectorXd of length Q containing ||dC/du|| at each point.
     */
    Eigen::VectorXd jacobian_det(const Eigen::MatrixXd& params) const override;

private:
    /// @brief Tensor product of the single 1D basis
    TensorProduct tensor_product_;

    /**
     * @brief Internal parametric derivative evaluation.
     *
     * result[k] is d^k C / du^k, shape (Q × gdim).
     */
    std::vector<Eigen::MatrixXd> eval_parametric_derivs_(const Eigen::MatrixXd& params,
                                                         std::size_t order) const;
};

} // namespace pyck

#endif // PYCK_CURVE_HPP
