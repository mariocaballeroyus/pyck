#ifndef PYCK_SURFACE_HPP
#define PYCK_SURFACE_HPP

#include <array>
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

class SurfacePatch : public Patch
{
public:

    // --- Constructors & Factory Methods ------------------------------------------------

    /**
     * @brief Construct a surface patch.
     * 
     * @param gdim  Geometric dimension (e.g. 3 for a surface in 3D space).
     * @param bases Basis functions in the u and v parametric directions.
     * @param control_points  Control-point matrix of shape (n_u · n_v) × gdim,
     *        ordered with v running fastest:
     *        P_{0,0}, P_{0,1}, …, P_{0,n_v-1}, P_{1,0}, …
     */
    SurfacePatch(std::size_t gdim,
                 std::array<Basis*, 2> bases,
                 const Eigen::MatrixXd& control_points)
        : Patch(gdim, 2, control_points),
          tensor_product_({
              std::shared_ptr<Basis>(std::shared_ptr<Basis>{}, bases[0]),
              std::shared_ptr<Basis>(std::shared_ptr<Basis>{}, bases[1])
          }) {}

    /**
     * @brief Create a flat plate S(u,v) = (L_u·u, L_v·v, 0).
     *
     * Control points are placed at Greville abscissae so that the linear
     * mapping is represented exactly by any polynomial degree.
     *
     * @param bases B-spline bases in the u and v directions.
     * @param L_u   Physical length in the u direction.
     * @param L_v   Physical length in the v direction.
     * @return A 3D SurfacePatch representing the flat plate.
     */
    static SurfacePatch flat_plate(std::array<BSpline*, 2> bases,
                                   double L_u, double L_v);

    /**
     * @brief Create a parabolic surface S(u,v) = (u, v, u² + v²).
     *
     * Uses quadratic Bernstein bases (knots = {0,0,0,1,1,1}) in both
     * directions. The required control points are derived analytically.
     *
     * @param bases Quadratic B-spline bases (degree 2, 3 basis functions each).
     * @return A 3D SurfacePatch representing the paraboloid.
     */
    static SurfacePatch paraboloid(std::array<BSpline*, 2> bases);

    // --- Properties ----------------------------------------------------------------

    /// @brief Get the basis in a given parametric direction (0 = u, 1 = v).
    const Basis& basis(std::size_t dir) const override { return tensor_product_.basis(dir); }

    // --- Evaluation ----------------------------------------------------------------

    /**
     * @brief Evaluate the surface at given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @return Position matrix of shape (Q × gdim), where row i is S(u_i, v_i).
     */
    Eigen::MatrixXd eval(const Eigen::MatrixXd& params) const;

    /**
     * @brief Evaluate physical (spatial) derivatives of basis functions
     *        in local tangent-plane coordinates.
     *
     * Builds a local orthonormal frame on the surface at each evaluation
     * point via Gram-Schmidt (e₁ aligned with ∂S/∂u, e₂ orthogonal in the
     * tangent plane).  Derivatives are w.r.t. arc-length–scaled local
     * coordinates (x, y) on the surface — NOT global Cartesian axes.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @param order   Maximum derivative order (1, 2, or 3). Default = 1.
     * @return An unordered map from derivative label to a (Q × n_basis) matrix.
     *
     *         Order 1 keys: "N", "dNdx", "dNdy"
     *         Order 2 adds: "dNdxdx", "dNdxdy", "dNdydy"
     *         Order 3 adds: "dNdxdxdx", "dNdxdxdy", "dNdxdydy", "dNdydydy"
     */
    std::unordered_map<std::string, Eigen::MatrixXd>
    eval_physical_derivs(const Eigen::MatrixXd& params,
                         std::size_t order = 1) const;

    /**
     * @brief Compute the Jacobian matrices at the given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @return A vector of Q matrices, each of shape (gdim × 2).
     *         Column 0 is ∂S/∂u, column 1 is ∂S/∂v.
     */
    std::vector<Eigen::MatrixXd> jacobian(const Eigen::MatrixXd& params) const override;

    /**
     * @brief Compute the Jacobian determinant at the given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @return VectorXd of length Q containing √det(J^T J) at each point.
     */
    Eigen::VectorXd jacobian_det(const Eigen::MatrixXd& params) const override;

private:
    /// @brief Tensor product of the two 1D bases
    TensorProduct tensor_product_;

    /**
     * @brief Internal parametric derivative evaluation.
     *
     * Same semantics as the old public eval_derivs: result[i][j] is
     * ∂^{i+j}S / (∂u^i ∂v^j), shape (Q × gdim).
     */
    std::vector<std::vector<Eigen::MatrixXd>>
    eval_parametric_derivs_(const Eigen::MatrixXd& params,
                            std::size_t order) const;
};

} // namespace pyck

#endif // PYCK_SURFACE_HPP