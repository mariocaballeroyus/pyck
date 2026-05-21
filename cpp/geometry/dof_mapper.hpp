#ifndef PYCK_DOF_MAPPER_HPP
#define PYCK_DOF_MAPPER_HPP

#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>

#include "../types.hpp"

namespace pyck
{

template <std::size_t d>
class DofMapper
{
public:

    /**
     * @brief Default-construct an empty DofMapper.
     */
    DofMapper() = default;

    /**
     * @brief Construct a new DofMapper object
     *
     * @param num_basis Array of the number of basis functions in each direction
     * @param degree Array of the polynomial degree in each direction
     */
    DofMapper(const std::array<Index, d>& num_basis,
              const std::array<Index, d>& degree)
        : num_basis_(num_basis), degree_(degree) {}

    /**
     * @brief Set the per-direction basis counts and degrees in place.
     *
     * Equivalent to assigning from a freshly-constructed `DofMapper(num_basis, degree)`.
     */
    void init(const std::array<Index, d>& num_basis,
              const std::array<Index, d>& degree)
    {
        num_basis_ = num_basis;
        degree_    = degree;
    }

    /**
     * @brief Map logical tensor-product indices to a flattened global DOF index.
     *
     * @param logical_idx Array of d logical indices (e.g. {i, j} for 2D)
     */
    Index to_global(const std::array<Index, d>& logical_idx) const;

    /**
     * @brief Map a flattened global DOF index back to logical tensor-product indices.
     *
     * @param global_idx Global DOF index.
     * @return Array of d logical indices.
     */
    std::array<Index, d> from_global(Index global_idx) const;


    /**
     * @brief Get the global DOF indices for a specific layer from the boundary.
     *
     * @param param_dim The boundary dimension.
     * @param at_start True for start (e.g. u=0), false for end (e.g. u=1).
     * @param layer_idx The layer index (0 for outermost, 1 for next, etc.).
     */
    std::vector<Index> get_layer_dofs(Index param_dim, bool at_start, Index layer_idx) const;

    /**
     * @brief Write active global DOF indices for an element into a caller-owned buffer.
     *
     * Decodes elem_idx into per-direction spans and forwards to the
     * span-overload. Prefer the span-overload when spans are known.
     *
     * @param elem_idx Flat element index.
     * @param out      Resized in place; pre-size in a workspace to avoid allocation.
     */
    void get_element_dofs(Index elem_idx, std::vector<Index>& out) const;

    /**
     * @brief Write active global DOF indices for an element into a caller-owned buffer.
     *
     * Ordering matches TensorProduct::eval_on_span's tensor-product column
     * layout (dim-0 outer, dim-(d-1) inner) so basis column k pairs with
     * control-point entry k.
     *
     * @param span_indices Per-direction knot-span indices.
     * @param out          Resized in place; pre-size in a workspace to avoid allocation.
     */
    void get_element_dofs(const std::array<Index, d>& span_indices,
                          std::vector<Index>& out) const;

    /**
     * @brief Increment a d-dimensional logical index to the next value in lexicographical order.
     *        (fastest varying index first).
     *
     * @param logical_idx The current logical index array to be incremented.
     * @return true if the index was successfully incremented, false if it wrapped around to zero.
     */
    bool next_logical_index(std::array<Index, d>& logical_idx) const;

    /**
     * @brief Get the number of basis functions in each direction.
     */
    const std::array<Index, d>& num_basis() const { return num_basis_; }

    /**
     * @brief Get the polynomial degree in each direction.
     */
    const std::array<Index, d>& degree() const { return degree_; }

    /**
     * @brief Get the number of parametric intervals (elements) for each dimension.
     */
    std::array<Index, d> num_intervals() const {
        std::array<Index, d> res;
        for (std::size_t i = 0; i < d; ++i) {
            res[i] = (num_basis_[i] > degree_[i]) ? (num_basis_[i] - degree_[i]) : 0;
        }
        return res;
    }

    /**
     * @brief Get the total number of DOF indices (product of num_basis).
     */
    std::size_t total_dofs() const {
        std::size_t total = 1;
        for (auto n : num_basis_) total *= n;
        return total;
    }

    /**
     * @brief Get the total number of elements.
     */
    std::size_t total_elements() const {
        std::size_t total = 1;
        for (auto n : num_intervals()) total *= n;
        return total;
    }

private:
    /// @brief Number of basis functions in each direction
    std::array<Index, d> num_basis_{};

    /// @brief Polynomial degree in each direction
    std::array<Index, d> degree_{};
};

} // namespace pyck

#endif // PYCK_DOF_MAPPER_HPP