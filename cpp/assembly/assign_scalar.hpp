#ifndef PYCK_ASSIGN_SCALAR_HPP
#define PYCK_ASSIGN_SCALAR_HPP

#include <vector>
#include <concepts>
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Strong enforcement of a global DOF value.
 *        Maintains the symmetry of the global stiffness matrix.
 *
 * @tparam T floating point type
 * @param dofs The global degrees of freedom to constrain
 * @param values The prescribed scalar values for each DOF. Must be same length as `dofs`.
 * @param stiffness The global stiffness matrix (will be modified).
 * @param load The global RHS load vector (will be modified).
 */
template <std::floating_point T>
void assign_scalar(const std::vector<Index>& dofs,
                   const std::vector<T>& values,
                   Matrix<T>& stiffness,
                   Vector<T>& load);

/**
 * @brief Strong enforcement of a global DOF value to 0.0.
 *        Maintains the symmetry of the global stiffness matrix.
 * 
 * @tparam T floating point type
 * @param dofs The global degrees of freedom to constrain to 0.0
 * @param stiffness The global stiffness matrix to modify.
 * @param load The global load vector to modify.
 */
template <std::floating_point T>
void assign_zeros(const std::vector<Index>& dofs,
                  Matrix<T>& stiffness,
                  Vector<T>& load);

} // namespace pyck

#endif // PYCK_ASSIGN_SCALAR_HPP
