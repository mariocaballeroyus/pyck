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
     * @brief Evaluate the non-zero tensor-product basis functions within a given
     *        multi-dimensional knot span.
     *
     * @param params A matrix of (m × d) parametric coordinates (all in the same span).
     * @param spans  Per-direction knot-span indices.
     * @return A matrix of size (m × K) where K = prod(p_i + 1).  The column ordering
     *         follows the same tensor-product layout as the 1D bases.
     */
    Matrix<T> eval(const Matrix<T>& params,
                   const std::array<Index, d>& spans) const;

    /**
     * @brief Evaluate non-zero basis functions and mixed partial derivatives for the
     *        tensor product space within a given multi-dimensional knot span.
     *
     * @param params A matrix of size (m × d) containing parametric points.
     * @param spans  Per-direction knot-span indices.
     * @param order  Maximum derivative order to compute (same for all directions).
     * @return A flat std::vector of matrices (one per derivative multi-index).
     */
    std::vector<Matrix<T>> eval_derivs(const Matrix<T>& params,
                                       const std::array<Index, d>& spans,
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