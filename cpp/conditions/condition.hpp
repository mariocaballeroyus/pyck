#ifndef PYCK_CONDITION_HPP
#define PYCK_CONDITION_HPP

#include <concepts>
#include <cstddef>

#include "patch.hpp"
#include "../assembly/dof_layout.hpp"
#include "../assembly/patch_blocks.hpp"
#include "../assembly/system_assembler.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Base interface for all assembly conditions.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Patch / element parametric dimension.
 */
template <std::floating_point T, std::size_t d>
class Condition
{
public:

    // === Constructors ===============================================================

    virtual ~Condition() = default;

    // === Utility ====================================================================

    /**
     * @brief Allocate auxiliary DOF blocks. Default: no-op.
     *
     * @param layout Equation-numbering authority (mutable).
     */
    virtual void allocate_dofs(DofLayout& layout) {}

    /**
     * @brief Assemble the condition's contribution into the global system.
     *
     * @param assembler  Sparse-assembly sink for stiffness and load (mutable).
     * @param layout     DOF layout (read-only).
     * @param blocks     Patch to primal-block resolver. A single-patch condition
     *                   resolves its own `patch()`; a coupling condition resolves
     *                   both of the patches it ties.
     */
    virtual void apply(SystemAssembler<T>& assembler,
                       const DofLayout& layout,
                       const PatchBlocks<T, d>& blocks) const = 0;

    // === Properties =================================================================

    /// @brief The patch the condition acts on. The assembler uses it to route the 
    ///        condition to its primal DOF block.
    virtual const Patch<T, d>& patch() const = 0;

};

} // namespace pyck

#endif // PYCK_CONDITION_HPP
