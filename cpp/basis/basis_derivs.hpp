#ifndef PYCK_BASIS_DERIVS_HPP
#define PYCK_BASIS_DERIVS_HPP

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

// === BasisDerivs wrapper ============================================================

/**
 * @brief Tensor-product shape functions and their derivatives on one element span,
 *        packed in the Gismo-style per-order layout.
 *
 * Storage (returned by @ref TensorProduct::eval_basis):
 *     data()[k]  is a (K · n_k) × Q col-major matrix, where
 *         K   = number of nonzero basis functions on the span
 *         n_k = number of upper-tri multi-indices of total order k = C(k+d-1, d-1)
 *         Q   = number of quadrature / evaluation points.
 *
 *     Row layout within data()[k]: for basis function `b ∈ [0, K)` and packed
 *     multi-index `m ∈ [0, n_k)`, ∂_m B_b(u_q) is at row `b · n_k + m`,
 *     column `q`. Multi-indices use Voigt order for k = 2 (diagonals first)
 *     and lex order for k = 3.
 *
 * Named accessors @ref N, @ref N_d1, @ref N_d2, @ref N_d3 return zero-copy
 * strided @c Eigen::Map views shaped (Q × K) to match the existing assembly
 * convention `M.row(q)` → all basis values at quadrature point q.
 */
template <std::floating_point T, std::size_t d>
class BasisDerivs
{
public:

    /// @brief Strided (Q × K) view into the packed per-order storage.
    using View = Eigen::Map<const Matrix<T>, Eigen::Unaligned,
                            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

    // === Constructors ===============================================================

    BasisDerivs() = default;

    /// @brief Wrap a per-order Gismo-layout buffer; takes ownership.
    explicit BasisDerivs(std::vector<Matrix<T>> per_order)
        : data_(std::move(per_order)) {}

    // === Properties =================================================================

    /// @brief Maximum derivative order present (0 if empty).
    Index order() const
    { return data_.empty() ? Index(0) : Index(data_.size()) - 1; }

    /// @brief Number of evaluation / quadrature points.
    Index Q() const { return data_[0].cols(); }

    /// @brief Number of nonzero basis functions on the span.
    Index K() const { return data_[0].rows(); }

    // === Named Accessors (Q × K) ====================================================

    /// @brief Shape function values. N(q, b) = B_b(u_q).
    View N() const { return view(0, 0); }

    /// @brief First derivative ∂_dir N. Element (q, b) = ∂_dir B_b(u_q).
    View N_d1(Index dir) const { return view(1, dir); }

    /// @brief Second derivative ∂_ij N (symmetric, canonicalised internally).
    View N_d2(Index i, Index j) const { return view(2, pack2<d>(i, j)); }

    /// @brief Third derivative ∂_ijk N (symmetric, canonicalised internally).
    View N_d3(Index i, Index j, Index k) const { return view(3, pack3<d>(i, j, k)); }

    // === Raw Access =================================================================

    /// @brief Underlying per-order packed buffers (Gismo layout).
    const std::vector<Matrix<T>>& data() const { return data_; }

private:

    /// @brief Build a (Q × K) strided view onto packed-position @p packed
    ///        within order-@p ord storage.
    View view(Index ord, Index packed) const
    {
        const Matrix<T>& m = data_[ord];
        const Index n_k = num_multi_indices<d>(ord);
        const Index k_  = m.rows() / n_k;
        const Index q_  = m.cols();
        return View(
            m.data() + packed,
            q_, k_,
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(n_k, n_k * k_));
    }

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
 * @return BasisDerivs wrapping the Gismo-layout per-order matrices.
 */
template <std::floating_point T, std::size_t d>
BasisDerivs<T, d>
eval_basis(const TensorProduct<T, d>& tp,
           const std::type_identity_t<ColMatrix<T, d>>& coords,
           const std::array<Index, d>& spans,
           Index order);

} // namespace pyck

#endif // PYCK_BASIS_DERIVS_HPP
