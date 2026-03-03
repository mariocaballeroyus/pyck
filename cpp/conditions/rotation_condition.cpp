#include "rotation_condition.hpp"
#include <stdexcept>
#include <cmath>

namespace pyck
{

template <std::floating_point T>
RotationCondition<T>::RotationCondition(std::vector<Index> dofs, T value)
{
    if (std::abs(value) > T(1e-14)) {
        throw std::invalid_argument(
            "RotationCondition currently supports only zero values. "
            "Non-zero prescribed rotations are not yet implemented.");
    }
    this->dofs_ = std::move(dofs);
}

template <std::floating_point T>
void RotationCondition<T>::apply(Matrix<T>& stiffness, Vector<T>& load) const
{
    assign_zeros<T>(this->dofs_, stiffness, load);
}

// === Template Instantiations ========================================================

template class RotationCondition<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class RotationCondition<float>;
#endif

} // namespace pyck
