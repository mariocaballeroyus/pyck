#ifndef PYCK_CONDITION_HPP
#define PYCK_CONDITION_HPP

#include <cstddef>
#include <vector>

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
     * @brief Apply the condition to the stiffness matrix and load vector
     *
     * @param stiffness Stiffness matrix
     * @param load Load vector
     */
    virtual void apply(Matrix<T>& stiffness, Vector<T>& load) const = 0;

    /**
     * @brief Number of additional Lagrange-multiplier DOFs the condition
     *        appends to the global system. Default 0 (penalty / load).
     */
    virtual std::size_t num_multipliers() const { return 0; }

    /**
     * @brief Set the global offset where this condition's multiplier rows
     *        start. The assembler calls this once before sizing K/F so that
     *        `apply()` can scatter into the correct rows/columns of the
     *        augmented system.
     */
    void set_multiplier_offset(std::size_t offset) { multiplier_offset_ = offset; }

    /// @brief Global offset of this condition's multiplier block.
    std::size_t multiplier_offset() const { return multiplier_offset_; }

    /**
     * @brief Set the degrees of freedom where the condition is applied
     *
     * @param dofs Degrees of freedom where the condition is applied
     */
    void set_dofs(std::vector<Index> dofs)
    { dofs_ = std::move(dofs); }

    /// @brief Get the degrees of freedom where the condition is applied
    const std::vector<Index>& dofs() const
    { return dofs_; }

protected:

    /// @brief Degrees of freedom where the condition is applied
    std::vector<Index> dofs_;

    /// @brief Global offset of the multiplier block (set by the assembler).
    std::size_t multiplier_offset_ = 0;

};

} // namespace pyck

#endif // PYCK_CONDITION_HPP
