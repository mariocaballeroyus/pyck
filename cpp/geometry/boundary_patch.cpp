#include "boundary_patch.hpp"
#include "patch.hpp"
#include "factories.hpp"
#include "factories.hpp"
#include "dof_mapper.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d> requires (d>1)
BoundaryPatch<T, d>::BoundaryPatch(const Ptr<Patch<T, d>>& parent,
                                   std::size_t param_dim,
                                   bool at_start)
    : Patch<T, d - 1>(parent->basis_ptr(1 - param_dim),
                    parent->get_control_points(parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0))),
      parent_(parent), param_dim_(param_dim), at_start_(at_start)
{
    if (param_dim >= d) {
        throw std::invalid_argument(
            "BoundaryPatch: param_dim is out of bounds for dimension d."
        );
    }

    parent_dofs_ = parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0);
}

// === Template Instantiations ========================================================

template class BoundaryPatch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BoundaryPatch<float, 2>;
#endif

} // namespace pyck
