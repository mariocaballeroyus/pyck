#ifndef PYCK_DIRECT_CONSTRAINT_HPP
#define PYCK_DIRECT_CONSTRAINT_HPP

#include <vector>
#include <concepts>
#include <stdexcept>

#include "constraint.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Strong enforcement of prescribed DOF values: `u[dofs[i]] = values[i]`.
 *
 *        Applies symmetric row-column elimination to the system of equations
 *        to enforce fixed displacement/boundary conditions.
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class DirectConstraint : public Constraint<T>
{
public:
    /**
     * @brief Construct a `DirectConstraint` from lists of DOF indices and 
     * values.
     *
     * @param dofs   Global DOF indices to constrain.
     * @param values Prescribed values.
     */
    DirectConstraint(std::vector<Index> dofs, 
                        std::vector<T> values);

    /**
     * @brief Construct a `DirectConstraint` from a list of DOF indices and a 
     * single value.
     *
     * @param dofs  Global DOF indices to constrain.
     * @param value Prescribed value for all DOFs.
     */
    DirectConstraint(std::vector<Index> dofs, 
                        T value = T(0));

    /**
     * @brief Apply the Dirichlet constraint to a stiffness matrix and load vector.
     *
     * @param stiffness Stiffness matrix
     * @param load      Load vector
     */
    void apply(Matrix<T>& stiffness, 
               Vector<T>& load) const override;

    /// @brief Get the constrained DOF indices
    const std::vector<Index>& dofs() const { return dofs_; }

    /// @brief Get the constrained DOF values
    const std::vector<T>& values() const { return values_; }

private:
    /// @brief Constrained DOF indices
    std::vector<Index> dofs_;

    /// @brief Constrained DOF values
    std::vector<T> values_;
};

} // namespace pyck

#endif // PYCK_DIRECT_CONSTRAINT_HPP
