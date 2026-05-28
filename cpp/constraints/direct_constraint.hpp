#ifndef PYCK_DIRECT_CONSTRAINT_HPP
#define PYCK_DIRECT_CONSTRAINT_HPP

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
     * @brief Construct a DirectConstraint from per-DOF indices and values.
     */
    DirectConstraint(IndexVector dofs,
                     Vector<T>   values);

    /**
     * @brief Construct a DirectConstraint from DOF indices and a single
     * value applied to all of them.
     */
    DirectConstraint(IndexVector dofs,
                     T value = T(0));

    /**
     * @brief Apply the Dirichlet constraint to a stiffness matrix and load vector.
     */
    void apply(SparseMatrix<T>& stiffness,
               Vector<T>& load) const override;

    /// @brief Get the constrained DOF indices
    const IndexVector& dofs() const { return dofs_; }

    /// @brief Get the constrained DOF values
    const Vector<T>& values() const { return values_; }

private:
    /// @brief Constrained DOF indices
    IndexVector dofs_;

    /// @brief Constrained DOF values
    Vector<T> values_;
};

} // namespace pyck

#endif // PYCK_DIRECT_CONSTRAINT_HPP
