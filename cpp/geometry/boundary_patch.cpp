#include "boundary_patch.hpp"
#include "patch.hpp"
#include "factories.hpp"
#include "factories.hpp"
#include "dof_mapper.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
BoundaryPatch<T, d>::BoundaryPatch(const Ptr<Patch<T, d>>& parent,
                                   std::size_t param_dim,
                                   bool at_start)
    : parent_(parent), param_dim_(param_dim), at_start_(at_start)
{
    if (param_dim >= d) {
        throw std::invalid_argument(
            "BoundaryPatch: param_dim is out of bounds for dimension d."
        );
    }

    const auto& mapper = parent_->dof_mapper();

    displacement_dofs_ = mapper.get_layer_dofs(param_dim, at_start, 0);
    rotation_dofs_     = mapper.get_layer_dofs(param_dim, at_start, 1);
    boundary_dofs_     = mapper.get_boundary_dofs(param_dim, at_start);
}

// === Template Instantiations ========================================================

template class BoundaryPatch<double, 1>;
template class BoundaryPatch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BoundaryPatch<float, 1>;
template class BoundaryPatch<float, 2>;
#endif

} // namespace pyck
