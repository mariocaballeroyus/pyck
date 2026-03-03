#ifndef PYCK_DISPLACEMENT_CONDITION_HPP
#define PYCK_DISPLACEMENT_CONDITION_HPP

#include <vector>

#include "condition.hpp"
#include "assign_scalar.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Homogeneous displacement boundary condition (prescribed zero displacement).
 *
 *        Stores a set of displacement DOF indices and enforces u_i = 0 for each one
 *        by calling assign_zeros() during apply().
 *
 *        Currently only zero values are supported. Passing a non-zero value
 *        to the constructor will throw std::invalid_argument.
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class DisplacementCondition : public Condition<T>
{
public:

    /**
     * @brief Construct a DisplacementCondition from a list of displacement DOF indices.
     *
     * @param dofs  Global displacement DOF indices to constrain.
     * @param value Prescribed value (must be zero; non-zero throws).
     */
    explicit DisplacementCondition(std::vector<Index> dofs, T value = T(0));

    /**
     * @brief Apply the homogeneous displacement condition to the stiffness matrix and load vector.
     */
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;
};

} // namespace pyck

#endif // PYCK_DISPLACEMENT_CONDITION_HPP
