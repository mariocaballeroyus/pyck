#ifndef PYCK_CONDITION_HPP
#define PYCK_CONDITION_HPP

#include <vector>

#include "../types.hpp"

namespace pyck
{

template <std::floating_point T>
class Condition
{
public:

    virtual ~Condition() = default;

    /// @brief Set the degrees of freedom where the condition is applied
    void set_dofs(std::vector<Index> dofs)
    { dofs_ = std::move(dofs); }

    /// @brief Get the degrees of freedom where the condition is applied
    const std::vector<Index>& dofs() const
    { return dofs_; }

    /**
     * @brief Apply the condition to the stiffness matrix and load vector
     *
     * @param stiffness Stiffness matrix
     * @param load Load vector
     */
    virtual void apply(Matrix<T>& stiffness, Vector<T>& load) const = 0;

protected:

    /// @brief Degrees of freedom where the condition is applied
    std::vector<Index> dofs_;

};

} // namespace pyck

#endif // PYCK_CONDITION_HPP