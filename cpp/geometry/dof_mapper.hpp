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

    explicit DofMapper(const std::array<Index, d>& num_basis)
        : num_basis_(num_basis) {}

    /**
     * @brief Map logical tensor-product indices to a flattened global DOF index.
     *
     * @param logical_idx Array of d logical indices (e.g. {i, j} for 2D)
     */
    Index to_global(const std::array<Index, d>& logical_idx) const;

    /**
     * @brief Get the global DOF indices for the fully clamped boundary layers.
     *
     * @param param_dim The boundary dimension (0 for u, 1 for v, etc.)
     * @param at_start If true, clamped at u=0. If false, clamped at u=1.
     */
    std::vector<Index> get_boundary_dofs(Index param_dim, bool at_start) const;

    /**
     * @brief Get the number of basis functions in each direction.
     */
    const std::array<Index, d>& num_basis() const { return num_basis_; }

private:
    std::array<Index, d> num_basis_;
};

} // namespace pyck

#endif // PYCK_DOF_MAPPER_HPP