#include "mixed_displacement_shell.hpp"

#include <stdexcept>
#include <utility>

namespace pyck
{

template <std::floating_point T>
MixedDisplacementShell<T>::MixedDisplacementShell(Ptr<Element<T, 2>> base)
    : base_(std::move(base))
{
    if (!base_) {
        throw std::invalid_argument("MixedDisplacementShell: base element is null.");
    }
}

// === Template Instantiations ========================================================

template class MixedDisplacementShell<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class MixedDisplacementShell<float>;
#endif

} // namespace pyck
