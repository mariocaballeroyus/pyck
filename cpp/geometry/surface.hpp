#ifndef PYCK_SURFACE_HPP
#define PYCK_SURFACE_HPP

#include <cstddef>
#include <array>
#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "patch.hpp"
#include "tensor.hpp"

namespace pyck
{

class SurfacePatch : public Patch
{
public:
    /**
     * @brief Construct a surface patch.
     * 
     * @param gdim Geometric dimension of the surface (e.g. 3 for a surface in 3D space).
     * @param basis_u Basis functions in the u parametric direction.
     * @param basis_v Basis functions in the v parametric direction.
     * @param control_points Control points stored as a matrix of shape 
     *        (n_u * n_v) x gdim, ordered with v running fastest (row-major
     *        tensor-product ordering): P_{0,0}, P_{0,1}, ..., P_{0,n_v-1},
     *        P_{1,0}, ...
     */
    SurfacePatch(std::size_t gdim,
                 Basis* basis_u,
                 Basis* basis_v,
                 const Eigen::MatrixXd& control_points)
        : Patch(gdim, 2, control_points),
          basis_{basis_u, basis_v},
          tensor_product_({
              // Non-owning shared_ptrs via aliasing constructor:
              // the empty shared_ptr owns nothing, the alias points to the raw ptr
              std::shared_ptr<Basis>(std::shared_ptr<Basis>{}, basis_u),
              std::shared_ptr<Basis>(std::shared_ptr<Basis>{}, basis_v)
          }) {}

    /// @brief Get the basis in a given parametric direction (0 = u, 1 = v)
    const Basis& basis(std::size_t dir) const override 
    { return *basis_[dir]; }

    /**
     * @brief Evaluate the surface and its partial derivatives at a tensor-product
     *        grid of parameter values.
     * 
     * Given vectors u (size M) and v (size N), the evaluation points form an
     * M x N grid.  Each output matrix has (M*N) rows (one per grid point,
     * with v running fastest) and `gdim` columns.
     * 
     * @param u  Parameter values in the u direction (size M).
     * @param v  Parameter values in the v direction (size N).
     * @param order  Maximum total derivative order to compute.
     * @return A 2-D vector indexed as result[i][j] for
     *         ∂^(i+j) S / ∂u^i ∂v^j , with 0 ≤ i+j ≤ order.
     *         Each matrix is (M*N) x gdim.
     */
    std::vector<std::vector<Eigen::MatrixXd>> eval(const Eigen::VectorXd& u,
                                                   const Eigen::VectorXd& v,
                                                   std::size_t order = 0) const;

    /**
     * @brief Compute the Jacobian matrix and its determinant at a
     *        tensor-product grid of parameter values.
     *
     * @param u  Parameter values in the u direction (size M).
     * @param v  Parameter values in the v direction (size N).
     * @return A pair:
     *         - first:  vector of M*N matrices, each (gdim × 2).
     *         - second: VectorXd of length M*N with the determinants.
     */
    std::pair<std::vector<Eigen::MatrixXd>, Eigen::VectorXd>
    jacobian(const Eigen::VectorXd& u, const Eigen::VectorXd& v) const override;

    /**
     * @brief Build the tensor-product basis matrices and their parametric
     *        derivatives at a tensor-product grid.
     *
     * @param u  Parameter values in the u direction (size M).
     * @param v  Parameter values in the v direction (size N).
     * @param order  Maximum total derivative order.
     * @return A 2-D vector indexed as result[i][j] for
     *         the (i,j)-derivative of the tensor-product basis.
     *         Each matrix is (M*N) × (n_u * n_v).
     */
    std::vector<std::vector<Eigen::MatrixXd>> tensor_basis(
        const Eigen::VectorXd& u,
        const Eigen::VectorXd& v,
        std::size_t order = 0) const;

private:
    /// @brief Basis functions in the u and v parametric directions
    std::array<Basis*, 2> basis_;

    /// @brief Tensor product of the two 1D bases
    TensorProduct tensor_product_;
};

} // namespace pyck

#endif // PYCK_SURFACE_HPP