#ifndef PYCK_TENSOR_PRODUCT_HPP
#define PYCK_TENSOR_PRODUCT_HPP

#include "basis.hpp"

namespace pyck
{

/**
 * @brief Tensor product basis functions defined on a multi-dimensional parametric space
 * 
 * This class represents a tensor product of 1D basis functions. It allows for the evaluation 
 * of multi-dimensional basis functions by taking the tensor product of the corresponding 1D 
 * basis functions defined on each parametric dimension.
 */
class TensorProduct 
{
public:
    /**
     * @brief Construct a tensor product basis from a vector of 1D bases
     * 
     * @param bases A vector of shared pointers to 1D basis objects. The `i`-th element corresponds 
     *              to the basis functions defined on the `i`-th parametric dimension.
     */
    explicit TensorProduct(std::vector<std::shared_ptr<Basis>> bases)
        : bases_(std::move(bases)) {}

    /**
     * @brief Evaluate the N-dimensional basis functions
     * @param params A matrix of 1 .. `m` parametric point coordinates in `dim` dimensions.
     * 
     *                 params = { u_0^1 u_0^2 ... u_0^dim
     *                            u_1^1 u_1^2 ... u_1^dim
     *                            ...   ...   ... ...
     *                            u_m^1 u_m^2 ... u_m^dim }
     * 
     * @return A matrix of size (m x k) where `k` is the product of num_basis() of all 1D bases.
     * 
     *         result = { R_0(u_0) R_1(u_0) ... R_{K-1}(u_0)
     *                    R_0(u_1) R_1(u_1) ... R_{K-1}(u_1)
     *                    ...      ...      ... ...
     *                    R_0(u_m) R_1(u_m) ... R_{K-1}(u_m) }
     */
    Eigen::MatrixXd eval(const Eigen::MatrixXd& params) const;

    /**
     * @brief Evaluate basis and mixed partial derivatives for the tensor product space
     * 
     * @param params A matrix of size (m x d) containing 'm' parametric points in 'd' dimensions.
     * Each row represents a single multi-dimensional point (e.g., u, v, w).
     * @param orders A vector of length 'd' specifying the maximum derivative order to compute 
     * for each respective parametric dimension.
     * 
     * @return A flat std::vector of Eigen::MatrixXd containing all combinations of partial 
     *         derivatives. The total number of matrices is the product of (orders[i] + 1) 
     *         for all dimensions i.
     * 
     *          The derivative combinations are unrolled into the vector in lexicographical 
     *          order, meaning the last parametric dimension varies the fastest.
     * 
     *          Example for a 2D surface (d=2) with orders = {1, 2}:
     *          The vector will contain (1+1) * (2+1) = 6 matrices at the following indices:
     *          - [0]: d^0 / (du^0 dv^0)  -> Basis values
     *          - [1]: d^1 / (du^0 dv^1)  -> 1st partial w.r.t v
     *          - [2]: d^2 / (du^0 dv^2)  -> 2nd partial w.r.t v
     *          - [3]: d^1 / (du^1 dv^0)  -> 1st partial w.r.t u
     *          - [4]: d^2 / (du^1 dv^1)  -> Mixed partial (1st u, 1st v)
     *          - [5]: d^3 / (du^1 dv^2)  -> Mixed partial (1st u, 2nd v)
     * 
     *          Each Eigen::MatrixXd in the returned vector has size (m x K), where 'K' is the 
     *          total number of tensor product basis functions (the product of num_basis() 
     *          from all constituent 1D bases). Row 'i' of a matrix contains the specific 
     *          derivative evaluated at the 'i'-th input point.
     *          
     *          result[i] = { d^0(u_0) d^0(u_1) ... d^0(u_m)
     *                        d^1(u_0) d^1(u_1) ... d^1(u_m)
     *                        ...      ...      ... ...
     *                        d^2(u_0) d^2(u_1) ... d^2(u_m) }
     */
    std::vector<Eigen::MatrixXd> eval_derivs(const Eigen::MatrixXd& params, 
                                             const std::vector<std::size_t>& orders) const;

    /// @brief Get the parametric dimension (e.g., 2 for a surface)
    std::size_t dim() const { return bases_.size(); }

    /// @brief Total number of basis functions (n_u * n_v * ...)
    std::size_t num_basis() const;

private:
    /// @brief Vector of shared pointers to 1D basis objects for each parametric dimension
    std::vector<std::shared_ptr<Basis>> bases_;
};

} // namespace pyck

#endif // PYCK_TENSOR_PRODUCT_HPP