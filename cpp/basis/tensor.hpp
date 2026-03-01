#ifndef PYCK_TENSOR_PRODUCT_HPP
#define PYCK_TENSOR_PRODUCT_HPP

#include <array>
#include <memory>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <concepts>

#include "basis.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Tensor product basis functions defined on a multi-dimensional parametric space
/// @tparam T Scalar type
/// @tparam d Parametric dimension
template <std::floating_point T, std::size_t d>
class TensorProduct 
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Construct a tensor product basis from an array of 1D basis pointers
     * 
     * @param bases An array of shared pointers to 1D basis objects.
     */
    explicit TensorProduct(std::array<std::shared_ptr<const Basis<T>>, d> bases)
        : bases_(std::move(bases)) {}

    /**
     * @brief Construct a tensor product basis dynamically from individual basis pointers
     */
    template <typename... Args>
    requires (sizeof...(Args) == d)
    explicit TensorProduct(std::shared_ptr<const Args>... bases)
        : bases_({std::static_pointer_cast<const Basis<T>>(bases)...}) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the N-dimensional basis functions
     * @param params A matrix of 1 .. `m` parametric point coordinates in `d` dimensions.
     * 
     *                 params = { u_0^1 u_0^2 ... u_0^d
     *                            u_1^1 u_1^2 ... u_1^d
     *                            ...   ...   ... ...
     *                            u_m^1 u_m^2 ... u_m^d }
     * 
     * @return A matrix of size (m x k) where `k` is the product of num_basis() of all 1D bases.
     * 
     *         result = { R_0(u_0) R_1(u_0) ... R_{K-1}(u_0)
     *                    R_0(u_1) R_1(u_1) ... R_{K-1}(u_1)
     *                    ...      ...      ... ...
     *                    R_0(u_m) R_1(u_m) ... R_{K-1}(u_m) }
     */
    Matrix<T> eval(const Matrix<T>& params) const;

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
     */
    std::vector<Matrix<T>> eval_derivs(const Matrix<T>& params, 
                                                           const std::array<Index, d>& orders) const;

    // === Properties =================================================================

    /// @brief Get the parametric dimension (e.g., 2 for a surface)
    static constexpr std::size_t dim() { return d; }

    /// @brief Total number of basis functions (n_u * n_v * ...)
    Index num_basis() const;

    /// @brief Get the 1D basis for a given parametric direction
    /// @tparam Dir Parametric direction index
    template <std::size_t Dir>
    const Basis<T>& basis() const { return *bases_[Dir]; }

    /// @brief Get the 1D basis for a given parametric direction (runtime index)
    const Basis<T>& basis(std::size_t dir) const { 
        if (dir >= d) throw std::out_of_range("Basis dimension index out of range");
        return *bases_[dir]; 
    }

private:

    // === Member Variables ===========================================================

    /// @brief Array of 1D basis objects for each parametric dimension
    std::array<std::shared_ptr<const Basis<T>>, d> bases_;
};

} // namespace pyck

#endif // PYCK_TENSOR_PRODUCT_HPP