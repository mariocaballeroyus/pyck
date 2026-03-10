#ifndef PYCK_STRONG_DIRICHLET_CONSTRAINT_HPP
#define PYCK_STRONG_DIRICHLET_CONSTRAINT_HPP

#include <vector>

#include "constraint.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Strong enforcement of Dirichlet boundary conditions (prescribed displacements/rotations).
 *
 *        Stores a set of DOF indices and their prescribed values, and enforces
 *        u_i = v_i for each one by calling assign_scalar() during apply().
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class StrongDirichletConstraint : public Constraint<T>
{
public:

    /**
     * @brief Construct a StrongDirichletConstraint from lists of DOF indices and values.
     *
     * @param dofs   Global DOF indices to constrain.
     * @param values Prescribed values.
     */
    StrongDirichletConstraint(std::vector<Index> dofs, std::vector<T> values);

    /**
     * @brief Construct a StrongDirichletConstraint from a list of DOF indices and a single value.
     *
     * @param dofs  Global DOF indices to constrain.
     * @param value Prescribed value for all DOFs.
     */
    StrongDirichletConstraint(std::vector<Index> dofs, T value = T(0));

    /**
     * @brief Apply the Dirichlet condition to the stiffness matrix and load vector.
     *
     * @param stiffness Stiffness matrix
     * @param load Load vector
     */
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;

    /// @brief Get the degrees of freedom where the condition is applied
    const std::vector<Index>& dofs() const { return dofs_; }

    /// @brief Get the prescribed values
    const std::vector<T>& values() const { return values_; }

private:
    std::vector<Index> dofs_;
    std::vector<T> values_;
};

} // namespace pyck

#endif // PYCK_STRONG_DIRICHLET_CONSTRAINT_HPP
