#ifndef PYCK_CONDITION_HPP
#define PYCK_CONDITION_HPP

#include <concepts>
#include <cstddef>

#include "patch.hpp"
#include "../assembly/dof_layout.hpp"
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
     * @param stiffness     Global stiffness matrix (mutable).
     * @param load          Global load vector (mutable).
     * @param layout        DOF layout (read-only).
     * @param primal_block  This condition's patch primal block.
     */
    virtual void apply(Matrix<T>& stiffness,
                       Vector<T>& load,
                       const DofLayout& layout,
                       DofLayout::BlockId primal_block) const = 0;

    // === Properties =================================================================

    /// @brief The patch the condition acts on. The assembler uses it to route the 
    ///        condition to its primal DOF block.
    virtual const Patch<T, d>& patch() const = 0;

};

} // namespace pyck

#endif // PYCK_CONDITION_HPP
