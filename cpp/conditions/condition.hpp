#ifndef PYCK_CONDITION_HPP
#define PYCK_CONDITION_HPP

#include <cstddef>
#include <vector>

#include "../assembly/dof_layout.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Base interface for all assembly conditions.
 *
 *        Conditions store the information they need (patch index, boundary,
 *        etc.) and expose a single `apply` entry point. The assembler hands
 *        them the global K, F, layout, and the full per-patch primal block
 *        vector; each condition picks out the blocks it touches.
 */
template <std::floating_point T>
class Condition
{
public:

    virtual ~Condition() = default;

    /**
     * @brief Allocate auxiliary DOF blocks. Default: no-op.
     *
     * @param layout         Equation-numbering authority (mutable).
     * @param primal_blocks  Per-patch primal block IDs in patch order.
     */
    virtual void allocate_dofs(DofLayout& layout,
                               const std::vector<DofLayout::BlockId>& primal_blocks) {}

    /**
     * @brief Assemble the condition's contribution into K and F.
     *
     * @param stiffness      Global stiffness matrix (mutable).
     * @param load           Global load vector (mutable).
     * @param layout         DOF layout (read-only).
     * @param primal_blocks  Per-patch primal block IDs in patch order.
     */
    virtual void apply(Matrix<T>& stiffness,
                       Vector<T>& load,
                       const DofLayout& layout,
                       const std::vector<DofLayout::BlockId>& primal_blocks) const = 0;

    /// @brief Assign the index of the patch this condition acts on.
    void set_patch_idx(std::size_t idx) { patch_idx_ = idx; }
    std::size_t patch_idx() const { return patch_idx_; }

protected:
    std::size_t patch_idx_ = 0;
};

} // namespace pyck

#endif // PYCK_CONDITION_HPP
