#ifndef PYCK_TENSOR_PRODUCT_HPP
#define PYCK_TENSOR_PRODUCT_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "basis.hpp"
#include "evaluation.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
class TensorProduct;

// === BasisValues ====================================================================

/**
 * @brief Tensor-product shape functions and their derivatives on one element span,
 *        packed in Gismo-style per-order storage.
 *
 * Storage (returned by @ref TensorProduct::eval_on_span):
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
 * kernels are written as q-outer loops reading from these slabs:
 *
 *     for (Index q = 0; q < Q; ++q) {
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
    const std::vector<Matrix<T>>& data() const { return data_; }

    /// @brief Mutable accessor; intended for fill kernels like
    ///        `TensorProduct::eval_on_span` writing into a caller-owned buffer.
    std::vector<Matrix<T>>& data() { return data_; }

    /**
     * @brief Resize the per-order matrices to hold N = K active basis functions
     *        evaluated at Q points up to derivative @p order.
     *
     * @details Eigen's `Matrix::resize(rows, cols)` is a no-op when the shape
     *          is unchanged, so reusing a `BasisValues` across calls with
     *          identical (K, Q, order) skips all allocation after the first.
     */
    void reset_for(Index K, Index Q, Index order)
    {
        data_.resize(order + 1);
        for (Index k = 0; k <= order; ++k) {
            Index n_k = 0;
            if constexpr (d == 1)      n_k = 1;
            else if constexpr (d == 2) n_k = k + 1;
            else if constexpr (d == 3) n_k = (k + 1) * (k + 2) / 2;
            data_[k].resize(K * n_k, Q);
        }
    }

private:

    /// @brief Storage: data()[k] is (N · n_k) × Q; row `b · n_k + m` and
    ///        column `q` holds ∂_m B_b(u_q).
    std::vector<Matrix<T>> data_;
};

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
     *        @p order on one element span, writing into a caller-owned
     *        `BasisValues` in the packed layout described in @ref BasisValues.
     *
     * @param coords (Q × d) parametric coordinates on the span.
     * @param spans  Per-direction knot-span indices.
     * @param order  Highest total derivative order to evaluate (0, 1, 2 or 3).
     * @param eval   Recurrence scratch workspace, shared with the 1D basis
     *               evaluators. Reused across calls to avoid per-call
     *               heap allocation of the scratch buffers.
     * @param out    Output buffer; resized in place via `reset_for` (no-op if
     *               shape unchanged). Reusing one `BasisValues` across an
     *               assembly loop makes the steady-state path allocation-free.
     *
     * @throws std::invalid_argument if @p order is outside [0, 3].
     */
    void
    eval_on_span(const std::type_identity_t<ColMatrix<T, d>>& coords,
                 const std::array<Index, d>& spans,
                 Index order,
                 Evaluator<T>& eval,
                 BasisValues<T, d>& out) const;

    /// @brief Convenience overload that constructs the output `BasisValues`
    ///        internally and returns it by move. Hot-path callers should pass
    ///        a reused output buffer to the primary overload above.
    BasisValues<T, d>
    eval_on_span(const std::type_identity_t<ColMatrix<T, d>>& coords,
                 const std::array<Index, d>& spans,
                 Index order,
                 Evaluator<T>& eval) const
    {
        BasisValues<T, d> out;
        eval_on_span(coords, spans, order, eval, out);
        return out;
    }

    /// @brief Convenience overload that constructs both the Evaluator and the
    ///        output `BasisValues` internally. One-shot callers only.
    BasisValues<T, d>
    eval_on_span(const std::type_identity_t<ColMatrix<T, d>>& coords,
                 const std::array<Index, d>& spans,
                 Index order) const
    {
        Evaluator<T> eval;
        return eval_on_span(coords, spans, order, eval);
    }

    /**
     * @brief Evaluate the tensor-product basis at @p Q points that may lie in
     *        **different** spans, returning the same packed layout as
     *        @ref eval_on_span.
     *
     * @details Per-direction knot spans are determined per point via
     *          `Basis::find_span`, so the caller need not pre-group points by
     *          element. Column `q` of the output carries the K = ∏(p_i+1)
     *          active basis values (and packed derivatives) at point
     *          `coords.row(q)`. K is constant, but the *identity* of the K
     *          active basis functions depends on the span — recover it from
     *          `Basis::find_span(coords(q, dim))` per direction.
     *
     *          A scratch `Evaluator` is created internally on each call.
     *          This is the FE-assembly entry point for scattered evaluation
     *          points; for hot per-element loops where all Q points share a
     *          span and a reusable `Evaluator` is available, call
     *          @ref eval_on_span directly instead.
     *
     * @param coords (Q × d) parametric coordinates (one point per row).
     * @param order  Highest total derivative order to evaluate (0, 1, 2 or 3).
     *
     * @throws std::invalid_argument if @p order is outside [0, 3].
     */
    BasisValues<T, d>
    eval(const std::type_identity_t<ColMatrix<T, d>>& coords,
         Index order) const;

    // === Properties =================================================================

    /// @brief Get the parametric dimension (e.g., 2 for a surface)
    static constexpr std::size_t dim() { return d; }

    /// @brief Total number of basis functions
    Index num_basis() const;

    /// @brief Get the number of parametric intervals (elements) for each dimension
    std::array<Index, d> num_intervals() const;

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
