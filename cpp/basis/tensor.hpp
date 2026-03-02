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

/**
 * @brief Tensor product basis functions defined on a multi-dimensional parametric 
 *        space
 * @tparam T Scalar type
 * @tparam d Parametric dimension
 */
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
    explicit TensorProduct(std::array<Ptr<const Basis<T>>, d> bases)
        : bases_(std::move(bases)) {}

    /// @brief Construct a tensor product basis for a 1D curve
    explicit TensorProduct(Ptr<const Basis<T>> b0)
    requires (d == 1)
        : bases_{std::move(b0)} {}

    /// @brief Construct a tensor product basis for a 2D surface
    TensorProduct(Ptr<const Basis<T>> b0, 
                  Ptr<const Basis<T>> b1)
    requires (d == 2)
        : bases_{std::move(b0), std::move(b1)} {}

    /// @brief Construct a tensor product basis for a 3D volume
    TensorProduct(Ptr<const Basis<T>> b0, 
                  Ptr<const Basis<T>> b1, 
                  Ptr<const Basis<T>> b2)
    requires (d == 3)
        : bases_{std::move(b0), std::move(b1), std::move(b2)} {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the N-dimensional basis functions
     * @param params A matrix of 1 .. `m` parametric point coordinates in `d` 
     * dimensions.
     * 
     *                 params = { u_0^1 u_0^2 ... u_0^d
     *                            u_1^1 u_1^2 ... u_1^d
     *                            ...   ...   ... ...
     *                            u_m^1 u_m^2 ... u_m^d }
     * 
     * @return A matrix of size (m x k) where `k` is the product of num_basis() of all 
     *         1D bases.
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
     * @param params A matrix of size (m x d) containing 'm' parametric points in 'd' 
     *        dimensions. Each row represents a single multi-dimensional point.
     * @param order Maximum derivative order to compute (same for all directions).
     * 
     * @return A flat std::vector of matrices containing all combinations of partial 
     *         derivatives. The total number of matrices is (order + 1)^d.
     */
    std::vector<Matrix<T>> eval_derivs(const Matrix<T>& params, 
                                       Index order) const;

    // === Properties =================================================================

    /// @brief Get the parametric dimension (e.g., 2 for a surface)
    static constexpr std::size_t dim() 
    { return d; }

    /// @brief Total number of basis functions
    Index num_basis() const;

    /// @brief Get the number of parametric intervals (elements) for each dimension
    std::array<Index, d> num_intervals() const;

    /// @brief Get the 1D basis for a given parametric direction (runtime index)
    const Basis<T>& basis(Index dir) const;

private:

    // === Member Variables ===========================================================

    /// @brief Array of 1D basis objects for each parametric dimension
    std::array<Ptr<const Basis<T>>, d> bases_;
};

} // namespace pyck

#endif // PYCK_TENSOR_PRODUCT_HPP