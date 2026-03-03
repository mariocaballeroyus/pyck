#ifndef PYCK_ROTATION_CONDITION_HPP
#define PYCK_ROTATION_CONDITION_HPP

#include <vector>

#include "condition.hpp"
#include "assign_scalar.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Homogeneous rotation boundary condition (prescribed zero rotation/slope).
 *
 *        Stores a set of rotation DOF indices and enforces θ_i = 0 for each one
 *        by calling assign_zeros() during apply().
 *
 *        Currently only zero values are supported. Passing a non-zero value
 *        to the constructor will throw std::invalid_argument.
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class RotationCondition : public Condition<T>
{
public:

    /**
     * @brief Construct a RotationCondition from a list of rotation DOF indices.
     *
     * @param dofs  Global rotation DOF indices to constrain.
     * @param value Prescribed value (must be zero; non-zero throws).
     */
    explicit RotationCondition(std::vector<Index> dofs, T value = T(0));

    /**
     * @brief Apply the homogeneous rotation condition to the stiffness matrix and load vector.
     */
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;
};

} // namespace pyck

#endif // PYCK_ROTATION_CONDITION_HPP
