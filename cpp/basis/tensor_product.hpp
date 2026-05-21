#ifndef PYCK_TENSOR_PRODUCT_HPP
#define PYCK_TENSOR_PRODUCT_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "basis.hpp"
#include "bspline_algorithms.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
class TensorProduct;

// === Packed basis-values buffer =====================================================

/**
 * @brief Resize a per-order packed buffer to hold `K` active basis functions
 *        evaluated at `Q` points up to derivative @p order.
 */
template <std::floating_point T, std::size_t d>
inline void
resize_basis_buffer(std::vector<Matrix<T>>& buffer,
                    Index K, Index Q, Index order)
{
    buffer.resize(order + 1);
    for (Index k = 0; k <= order; ++k) {
        buffer[k].resize(K * num_multi_indices<d>(k), Q);
    }
}

// === TensorProduct ==================================================================

/**
 * @brief Tensor product basis functions defined on a multi-dimensional parametric
 *        space.
 *
 * @tparam T Scalar type.
 * @tparam d Parametric dimension.
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
     * @brief Evaluate the tensor-product basis and its derivatives up to total
     *        @p order on one element span.
     *
     * @param coords      (Q × d) parametric coordinates on the span.
     * @param spans       Per-direction knot-span indices.
     * @param order       Highest total derivative order to evaluate (0, 1, 2 or 3).
     * @param uni_results Per-direction univariate values + derivatives.
     * @param results     Output buffer (per-order packed).
     */
    void
    eval_on_span(const ColMatrix<T, d>& coords,
                 const std::array<Index, d>& spans,
                 Index order,
                 std::array<std::vector<Matrix<T>>, d>& uni_results,
                 std::vector<Matrix<T>>& results) const;

    /**
     * @brief Evaluate the tensor-product basis at @p Q points that may lie in
     *        different spans.
     * 
     * @param coords (Q × d) parametric coordinates (one point per row).
     * @param order  Highest total derivative order to evaluate (0, 1, 2 or 3).
     * @returns The packed layout with the evaluated values.
     */
    std::vector<Matrix<T>>
    eval_all(const ColMatrix<T, d>& coords, Index order) const;

    // === Properties =================================================================

    /// @brief Get the parametric dimension (e.g., 2 for a surface)
    static constexpr std::size_t dim() { return d; }

    /// @brief Total number of basis functions
    Index num_basis() const;

    /// @brief Get the number of parametric intervals (elements) for each dimension
    std::array<Index, d> num_intervals() const;

    /// @brief Total number of non-zero-volume elements in this tensor product.
    Index num_elements() const;

    /**
     * @brief Decode a live-element index into per-direction knot-span indices.
     *
     * @param live_idx Live element index. No bounds check; the caller is
     *                 responsible for staying in `[0, num_elements())`.
     */
    std::array<Index, d> decode_element(Index live_idx) const;

    /// @brief Get the 1D basis for a given parametric direction (runtime index)
    const Basis<T>& basis(Index dir) const;

    /// @brief Get a shared pointer to the 1D basis for a given parametric direction
    Ptr<const Basis<T>> basis_ptr(std::size_t dir) const { return bases_[dir]; }

private:

    /// @brief Array of 1D basis objects for each parametric dimension
    std::array<Ptr<const Basis<T>>, d> bases_;

};

} // namespace pyck

#endif // PYCK_TENSOR_PRODUCT_HPP
