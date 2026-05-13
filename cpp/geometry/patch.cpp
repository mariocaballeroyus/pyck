#include "patch.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   const ColMatrix<T, 3>& control_pts) requires(d == 1)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u)),
      dof_mapper_({tensor_product_.basis(0).num_basis()},
                  {tensor_product_.basis(0).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Control points must be embedded in 3D space.");
    }

    const Index expected_n = this->tensor_product_.basis(0).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Dimension mismatch.");
    }
}

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   Ptr<const Basis<T>> basis_v,
                   const ColMatrix<T, 3>& control_pts) requires(d == 2)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u), std::move(basis_v)),
      dof_mapper_({tensor_product_.basis(0).num_basis(), tensor_product_.basis(1).num_basis()},
                  {tensor_product_.basis(0).degree(),    tensor_product_.basis(1).degree()})
{
    if (control_pts.cols() != 3) 
    {
        throw std::invalid_argument("Patch<T, 2>: "
                                    "Control points must be embedded in 3D space.");
    }

    const Index expected_n = this->tensor_product_.basis(0).num_basis()
                           * this->tensor_product_.basis(1).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n) 
    {
        throw std::invalid_argument("Patch<T, 2>: "
                                    "Dimension mismatch.");
    }
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::active_control_pts(const std::array<Index, d>& spans) const
{
    auto dofs = dof_mapper_.get_element_dofs(spans);
    ColMatrix<T, 3> pts(dofs.size(), 3);
    for (Index i = 0; i < dofs.size(); ++i) {
        pts.row(i) = control_pts_.row(dofs[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::get_control_points(const std::vector<Index>& indices) const
{
    ColMatrix<T, 3> pts(indices.size(), 3);

    for (Index i = 0; i < indices.size(); ++i) 
    {
        pts.row(i) = control_pts_.row(indices[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
std::vector<Index>
Patch<T, d>::layer_dofs(std::size_t param_dim, bool at_start, Index layer_idx) const
{
    return dof_mapper_.get_layer_dofs(param_dim, at_start, layer_idx);
}

template <std::floating_point T, std::size_t d>
std::array<Index, d>
Patch<T, d>::decode_span(Index flat_idx) const
{
    if constexpr (d == 1) 
    {
        return {flat_idx};
    } 
    else
    {
        auto intervals = tensor_product_.num_intervals();
        std::array<Index, d> spans;
        Index temp = flat_idx;
        
        for (std::size_t i = 0; i < d; ++i) 
        {
            spans[i] = temp % intervals[i];
            temp /= intervals[i];
        }
        return spans;
    }
}

template <std::floating_point T, std::size_t d>
std::vector<Index>
Patch<T, d>::assembly_dofs() const
{
    std::vector<Index> indices(num_control_pts());
    for (Index i = 0; i < indices.size(); ++i) indices[i] = i;
    return indices;
}

// === Template Specializations =======================================================

template class Patch<double, 1>;
template class Patch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 1>;
template class Patch<float, 2>;
#endif

} // namespace pyck
