#ifndef PYCK_BASIS_VALUES_HPP
#define PYCK_BASIS_VALUES_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "../multi_index.hpp"
#include "../types.hpp"
#include "tensor.hpp"

namespace pyck
{

// === BasisValues ====================================================================

/**
 * @brief Tensor-product shape functions and their derivatives on one element span,
 *        packed in Gismo-style per-order storage.
 *
 * Storage (returned by @ref TensorProduct::eval_basis):
 *     data()[k]  is an (N · n_k) × Q col-major matrix, where
 *         N   = number of nonzero basis functions on the span
 *         n_k = number of upper-tri multi-indices of total order k
 *               = C(k + d - 1, d - 1)
 *         Q   = number of quadrature / evaluation points.
 *
 *     Row layout within data()[k]: for basis function `b ∈ [0, N)` and packed
 *     multi-index `m ∈ [0, n_k)`, ∂_m B_b(u_q) is at row `b · n_k + m`,
 *     column `q`. Multi-indices use Voigt order for k = 2 (diagonals first)
 *     and lex order for k = 3.
 *
 * Consumer usage: column `q` of data()[k] is a contiguous (N · n_k)-length
 * slab containing every basis-and-derivative at one quadrature point. Element
 * kernels are written as q-major loops reading from these slabs:
 *
 *     for (Index q = 0; q < N; ++q) {
 *         auto slab1 = basis.data()[1].col(q);  // contiguous (N · d)-length
 *         auto slab2 = basis.data()[2].col(q);  // contiguous (N · n_2)-length
 *         for (Index b = 0; b < N; ++b) {
 *             const T N_u_b  = slab1(b * d + 0);
 *             const T N_v_b  = slab1(b * d + 1);
 *             const T N_uu_b = slab2(b * n_2 + 0);    // Voigt: (0,0)
 *             // ...
 *         }
 *     }
 */
template <std::floating_point T, std::size_t d>
class BasisValues
{
public:

    // === Constructors ===============================================================

    BasisValues() = default;

    /// @brief Wrap a per-order packed storage; takes ownership.
    explicit BasisValues(std::vector<Matrix<T>> per_order)
        : data_(std::move(per_order)) {}

    // === Properties =================================================================

    /// @brief Maximum derivative order present (0 if empty).
    Index order() const
    { return data_.empty() ? Index(0) : Index(data_.size()) - 1; }

    /// @brief Number of evaluation / quadrature points.
    Index Q() const { return data_[0].cols(); }

    /// @brief Number of nonzero basis functions on the span.
    Index N() const { return data_[0].rows(); }

    // === Raw Access =================================================================

    /// @brief Vector of per-order packed matrices (one per derivative order).
    const std::vector<Matrix<T>>& data() const
    { return data_; }

private:

    /// @brief Storage: data()[k] is (N · n_k) × Q; row `b · n_k + m` and
    ///        column `q` holds ∂_m B_b(u_q).
    std::vector<Matrix<T>> data_;
};

// === Free Function ==================================================================

/**
 * @brief Evaluate basis functions and derivatives of a tensor product up to
 *        the given total order on one knot span.
 *
 * @param tp     Tensor-product basis.
 * @param coords (Q × d) parametric coordinates on the span.
 * @param spans  Per-direction knot-span indices.
 * @param order  Highest total derivative order to evaluate (0, 1, 2 or 3).
 * @return BasisValues wrapping the per-order packed matrices.
 */
template <std::floating_point T, std::size_t d>
BasisValues<T, d>
eval_basis(const TensorProduct<T, d>& tp,
           const std::type_identity_t<ColMatrix<T, d>>& coords,
           const std::array<Index, d>& spans,
           Index order);

} // namespace pyck

#endif // PYCK_BASIS_VALUES_HPP
