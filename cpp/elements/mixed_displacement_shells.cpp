#include "mixed_displacement_shells.hpp"

namespace pyck
{

// === Template Instantiations ========================================================

template class ShellReissnerMindlinHier4pMD<double>;
template class ShellReissnerMindlinHier5pMD<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlinHier4pMD<float>;
template class ShellReissnerMindlinHier5pMD<float>;
#endif

} // namespace pyck
