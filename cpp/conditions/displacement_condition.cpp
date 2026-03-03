#include "displacement_condition.hpp"
#include <stdexcept>
#include <cmath>

namespace pyck
{

template <std::floating_point T>
DisplacementCondition<T>::DisplacementCondition(std::vector<Index> dofs, T value)
{
    if (std::abs(value) > T(1e-14)) {
        throw std::invalid_argument(
            "DisplacementCondition currently supports only zero values. "
            "Non-zero prescribed displacements are not yet implemented.");
    }
    this->dofs_ = std::move(dofs);
}

template <std::floating_point T>
void DisplacementCondition<T>::apply(Matrix<T>& stiffness, Vector<T>& load) const
{
    assign_zeros<T>(this->dofs_, stiffness, load);
}

// === Template Instantiations ========================================================

template class DisplacementCondition<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class DisplacementCondition<float>;
#endif

} // namespace pyck
