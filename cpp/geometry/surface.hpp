#ifndef PYCK_SURFACE_HPP
#define PYCK_SURFACE_HPP

#include <array>
#include <cstddef>
#include <memory>
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
    const Basis& basis(std::size_t dir) const override 
    { return tensor_product_.basis(dir); }

    // --- Evaluation ----------------------------------------------------------------

    /**
     * @brief Evaluate the surface at given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @return Position matrix of shape (Q × gdim), where row i is S(u_i, v_i).
     */
    Eigen::MatrixXd eval(const Eigen::MatrixXd& params) const;

    /**
     * @brief Evaluate the surface and its partial derivatives.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @param order   Maximum total derivative order to compute.
     * @return A triangular 2-D vector indexed as result[i][j] for the
     *         mixed partial derivative ∂^{i+j} S / (∂u^i ∂v^j),
     *         with 0 ≤ i + j ≤ order. Each entry is a matrix of shape (Q × gdim).
     *
     *         For order = 1, the layout is:
     *           result[0][0] = S          (Q × gdim)
     *           result[0][1] = ∂S/∂v      (Q × gdim)
     *           result[1][0] = ∂S/∂u      (Q × gdim)
     *
     *         For order = 2, the layout adds:
     *           result[0][2] = ∂²S/∂v²    (Q × gdim)
     *           result[1][1] = ∂²S/∂u∂v   (Q × gdim)
     *           result[2][0] = ∂²S/∂u²    (Q × gdim)
     */
    std::vector<std::vector<Eigen::MatrixXd>> eval_derivs(
        const Eigen::MatrixXd& params, std::size_t order) const;

    /**
     * @brief Compute the Jacobian matrices at the given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @return A vector of Q matrices, each of shape (gdim × 2).
     *         Column 0 is ∂S/∂u, column 1 is ∂S/∂v.
     */
    std::vector<Eigen::MatrixXd>
    jacobian(const Eigen::MatrixXd& params) const override;

    /**
     * @brief Compute the Jacobian determinant at the given parametric points.
     *
     * @param params  Evaluation points, shape (Q × 2). Each row is a (u, v) pair.
     * @return VectorXd of length Q containing √det(J^T J) at each point.
     */
    Eigen::VectorXd
    jacobian_det(const Eigen::MatrixXd& params) const override;

private:
    /// @brief Tensor product of the two 1D bases
    TensorProduct tensor_product_;
};

} // namespace pyck

#endif // PYCK_SURFACE_HPP