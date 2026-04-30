#ifndef PYCK_CONDITION_HPP
#define PYCK_CONDITION_HPP

#include <cstddef>
#include <vector>

#include "../assembly/dof_layout.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T>
class Condition
{
public:

    /**
     * @brief Virtual destructor
     */
    virtual ~Condition() = default;

    /**
     * @brief Report how many DOFs this condition requires.
     *
     * The assembler queries this before sizing the global system, allocates
     * the DOFs, and provides the block IDs to apply().
     *
     * Default: 0 (no auxiliary DOFs).
     */
    virtual std::size_t num_dofs() const { return 0; }

    /**
     * @brief Allocate DOF blocks in the layout (if needed).
     *
     * Called by the assembler during the DOF allocation phase. Conditions
     * that require auxiliary DOFs override this to allocate their blocks.
     *
     * @param layout       Layout to allocate blocks against.
     * @param primal_block Handle to the primal DOF block.
     *
     * Default: no-op.
     */
    virtual void allocate_dofs(DofLayout& layout, DofLayout::BlockId primal_block) { }

    /**
     * @brief Apply the condition to the stiffness matrix and load vector.
     *
     * Called after K/F have been sized. The condition receives the primal block
     * ID so it can scatter DOFs appropriately.
     *
     * @param stiffness    Stiffness matrix
     * @param load         Load vector
     * @param layout       Equation-numbering authority for the assembly.
     * @param primal_block Handle to the primal DOF block.
     */
    virtual void apply(Matrix<T>& stiffness,
                       Vector<T>& load,
                       const DofLayout& layout,
                       DofLayout::BlockId primal_block) const = 0;

    /**
     * @brief Set the degrees of freedom where the condition is applied.
     *
     * Used by load-style conditions to record their target rows.
     */
    void set_dofs(std::vector<Index> dofs)
    { dofs_ = std::move(dofs); }

    /**
     * @brief Get the degrees of freedom where the condition is applied.
     */
    const std::vector<Index>& dofs() const
    { return dofs_; }

protected:

    /// @brief Degrees of freedom where the condition is applied
    std::vector<Index> dofs_;

};

} // namespace pyck

#endif // PYCK_CONDITION_HPP
