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
 *
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
    explicit TensorProduct(std::array<Ptr<const Basis<T>>, d> bases);

    // === Evaluation =================================================================

    /**
     * @brief Evaluate non-zero basis functions and mixed partial derivatives for the
     *        tensor product space within a given multi-dimensional knot span.
     *
     * @param params  A matrix of size (m × d) containing parametric points.
     * @param spans   Per-direction knot-span indices.
     * @param order   Maximum derivative order to compute (same for all directions).
     * @param results Output buffer; the function resizes it to (order+1)^d matrices.
     */
    void eval_on_span(const Matrix<T>& params,
                      const std::array<Index, d>& spans,
                      Index order,
                      std::vector<Matrix<T>>& results) const;

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

    /// @brief Get a shared pointer to the 1D basis for a given parametric direction
    Ptr<const Basis<T>> basis_ptr(std::size_t dir) const 
    { return bases_[dir]; }

private:

    /// @brief Array of 1D basis objects for each parametric dimension
    std::array<Ptr<const Basis<T>>, d> bases_;
    
};

} // namespace pyck

#endif // PYCK_TENSOR_PRODUCT_HPP