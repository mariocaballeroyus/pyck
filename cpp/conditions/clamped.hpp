#ifndef PYCK_CLAMPED_HPP
#define PYCK_CLAMPED_HPP

#include <vector>
#include <concepts>
#include "../types.hpp"
#include "../geometry/patch.hpp"

namespace pyck
{

namespace conditions
{

/// @brief Apply fully clamped boundary conditions (zero displacement and zero rotation)
/// @tparam T floating point type
/// @param dofs The degrees of freedom to fully clamp to 0.0
/// @param stiffness The global stiffness matrix to modify.
/// @param load The global load vector to modify.
template <std::floating_point T>
void apply_clamped_bc(const std::vector<Index>& dofs,
                      Matrix<T>& stiffness,
                      Vector<T>& load);

} // namespace conditions

} // namespace pyck

#endif // PYCK_CLAMPED_HPP
