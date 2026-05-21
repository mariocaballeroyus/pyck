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
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
class TensorProduct;

// === Packed basis-values buffer =====================================================

/**
 * @brief Tensor-product shape functions and their derivatives on one element
 *        span, packed in Gismo-style per-order storage as a plain
 *        `std::vector<Matrix<T>>`.
 *
 * Storage (produced by @ref TensorProduct::eval_on_span and
 *          @ref TensorProduct::eval):
 *     `buf[k]` is an (N · n_k) × Q col-major matrix, where
 *         N   = number of nonzero basis functions per evaluation point
 *               (= ∏(p_i + 1) on a tensor-product B-spline)
 *         n_k = number of upper-tri multi-indices of total order k
 *               = C(k + d - 1, d - 1)
 *         Q   = number of quadrature / evaluation points.
 *
 *     Row layout within `buf[k]`: for basis function `b ∈ [0, N)` and packed
 *     multi-index `m ∈ [0, n_k)`, ∂_m B_b(u_q) is at row `b · n_k + m`,
 *     column `q`. Multi-indices use Voigt order for k = 2 (diagonals first)
 *     and lex order for k = 3.
 *
 * Consumer usage: column `q` of `buf[k]` is a contiguous (N · n_k)-length
 * slab containing every basis-and-derivative at one quadrature point. Element
 * kernels are written as q-outer loops reading from these slabs:
 *
 *     for (Index q = 0; q < Q; ++q) {
 *         auto slab1 = basis[1].col(q);  // contiguous (N · d)-length
 *         auto slab2 = basis[2].col(q);  // contiguous (N · n_2)-length
 *         for (Index b = 0; b < N; ++b) {
 *             const T N_u_b  = slab1(b * d + 0);
 *             const T N_v_b  = slab1(b * d + 1);
 *             const T N_uu_b = slab2(b * n_2 + 0);    // Voigt: (0,0)
 *             // ...
 *         }
 *     }
 */

/**
 * @brief Resize a per-order packed buffer to hold `K` active basis functions
 *        evaluated at `Q` points up to derivative @p order.
 *
 * @details Eigen's `Matrix::resize` is a no-op when shape is unchanged, so
 *          reusing the same buffer across calls with identical (K, Q, order)
 *          is allocation-free after warm-up.
 */
template <std::floating_point T, std::size_t d>
inline void
resize_basis_buffer(std::vector<Matrix<T>>& buf,
                    Index K, Index Q, Index order)
{
    buf.resize(order + 1);
    for (Index k = 0; k <= order; ++k) {
        buf[k].resize(K * num_multi_indices<d>(k), Q);
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
     *        @p order on one element span, writing into a caller-owned
     *        per-order packed buffer (see the layout description above).
     *
     * @param coords (Q × d) parametric coordinates on the span.
     * @param spans  Per-direction knot-span indices.
     * @param order  Highest total derivative order to evaluate (0, 1, 2 or 3).
     * @param eval   Recurrence scratch workspace, shared with the 1D basis
     *               evaluators. Reused across calls to avoid per-call
     *               heap allocation of the scratch buffers.
     * @param uni    Per-direction univariate values + derivatives scratch.
     *               `uni[dim][k]` is `(p_dim+1) × Q`, populated by the 1D
     *               `Basis::eval_on_span` and consumed by the tensor-product
     *               nest. Pre-sizing in the caller (see `PatchValues`) keeps
     *               this path allocation-free.
     * @param out    Output buffer; resized in place via `resize_basis_buffer`
     *               (no-op if shape unchanged). Reusing one buffer across an
     *               assembly loop makes the steady-state path allocation-free.
     *
     * @throws std::invalid_argument if @p order is outside [0, 3].
     */
    void
    eval_on_span(const std::type_identity_t<ColMatrix<T, d>>& coords,
                 const std::array<Index, d>& spans,
                 Index order,
                 Evaluator<T>& eval,
                 std::array<std::vector<Matrix<T>>, d>& uni,
                 std::vector<Matrix<T>>& out) const;

    /// @brief Convenience overload that constructs the univariate scratch and
    ///        output buffer internally and returns the latter by move.
    ///        Hot-path callers should pass reused buffers to the primary
    ///        overload above.
    std::vector<Matrix<T>>
    eval_on_span(const std::type_identity_t<ColMatrix<T, d>>& coords,
                 const std::array<Index, d>& spans,
                 Index order,
                 Evaluator<T>& eval) const
    {
        std::array<std::vector<Matrix<T>>, d> uni;
        std::vector<Matrix<T>> out;
        eval_on_span(coords, spans, order, eval, uni, out);
        return out;
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
    std::vector<Matrix<T>>
    eval(const std::type_identity_t<ColMatrix<T, d>>& coords,
         Index order) const;

    // === Properties =================================================================

    /// @brief Get the parametric dimension (e.g., 2 for a surface)
    static constexpr std::size_t dim() { return d; }

    /// @brief Total number of basis functions
    Index num_basis() const;

    /// @brief Get the number of parametric intervals (elements) for each dimension
    std::array<Index, d> num_intervals() const;

    /**
     * @brief Total number of non-zero-volume elements in this tensor product.
     *
     * @details Equal to ∏_i (number of non-zero spans in `basis(i)`). Use this
     *          as the loop bound for live-element enumeration:
     *          `for (Index e = 0; e < tp.num_elements(); ++e) ...`. This skips
     *          the zero-width clamped spans that `num_intervals()` includes.
     */
    Index num_elements() const;

    /**
     * @brief Decode a live-element index into per-direction knot-span indices.
     *
     * @details Maps a live index `live_idx ∈ [0, num_elements())` to the
     *          tuple of underlying knot-span indices (one per parametric
     *          direction), all of which point at non-zero-width spans by
     *          construction. The unflattening uses U-inner ordering, matching
     *          the convention used by `DofMapper` and `decode_span`.
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
