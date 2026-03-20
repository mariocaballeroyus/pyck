#include "dof_mapper.hpp"
#include <algorithm>

namespace pyck
{

template <std::size_t d>
Index DofMapper<d>::to_global(const std::array<Index, d>& logical_idx) const
{
    Index global_idx = 0;
    Index multiplier = 1;
    for (std::size_t i = 0; i < d; ++i)
    {
        global_idx += logical_idx[i] * multiplier;
        multiplier *= num_basis_[i];
    }
    return global_idx;
}

template <std::size_t d>
std::array<Index, d> DofMapper<d>::from_global(Index global_idx) const
{
    std::array<Index, d> logical_idx;
    Index temp = global_idx;
    for (std::size_t i = 0; i < d; ++i)
    {
        logical_idx[i] = temp % num_basis_[i];
        temp /= num_basis_[i];
    }
    return logical_idx;
}

template <std::size_t d>
std::vector<Index> DofMapper<d>::get_boundary_dofs(std::size_t param_dim, 
                                                   bool at_start) const
{    
    if (param_dim >= d) {
        throw std::invalid_argument(
            "param_dim is out of bounds for the dimension d."
        );
    }

    std::vector<Index> dofs;
    std::array<Index, d> current_idx;
    current_idx.fill(0);
    
    // Only iterate over exactly the indices we need
    auto iterate = [&](auto& self, std::size_t current_dim) -> void {
        if (current_dim == d) {
            dofs.push_back(to_global(current_idx));
            return;
        }

        if (current_dim == param_dim) 
        {
            // Only loop over the two clamped boundary layers!
            if (at_start) {
                current_idx[current_dim] = 0;
                self(self, current_dim + 1);
                current_idx[current_dim] = 1;
                self(self, current_dim + 1);
            } else {
                current_idx[current_dim] = num_basis_[current_dim] - 2;
                self(self, current_dim + 1);
                current_idx[current_dim] = num_basis_[current_dim] - 1;
                self(self, current_dim + 1);
            }
        } 
        else 
        {
            // Full sweep for orthogonal dimensions
            for (Index i = 0; i < num_basis_[current_dim]; ++i) {
                current_idx[current_dim] = i;
                self(self, current_dim + 1);
            }
        }
    };

    iterate(iterate, 0);
    
    std::sort(dofs.begin(), dofs.end());
    
    return dofs;
}

template <std::size_t d>
std::vector<Index> DofMapper<d>::get_layer_dofs(std::size_t param_dim,
                                                bool at_start,
                                                Index layer_idx) const
{
    if (param_dim >= d) {
        throw std::invalid_argument(
            "param_dim is out of bounds for the dimension d."
        );
    }

    if (layer_idx >= num_basis_[param_dim]) {
        throw std::invalid_argument(
            "layer_idx is out of bounds for the number of basis functions."
        );
    }

    std::vector<Index> dofs;
    std::array<Index, d> current_idx;
    current_idx.fill(0);

    auto iterate = [&](auto& self, std::size_t current_dim) -> void {
        if (current_dim == d) {
            dofs.push_back(to_global(current_idx));
            return;
        }

        if (current_dim == param_dim) {
            // Fix index to the requested layer
            current_idx[current_dim] = at_start ? layer_idx : num_basis_[current_dim] - 1 - layer_idx;
            self(self, current_dim + 1);
        } else {
            for (Index i = 0; i < num_basis_[current_dim]; ++i) {
                current_idx[current_dim] = i;
                self(self, current_dim + 1);
            }
        }
    };

    iterate(iterate, 0);
    std::sort(dofs.begin(), dofs.end());
    return dofs;
}

template <std::size_t d>
std::vector<Index> DofMapper<d>::get_element_dofs(Index elem_idx) const
{
    // Compute the number of knot-span intervals per direction:
    //   intervals[i] = num_basis_[i] + degree_[i]
    std::array<Index, d> intervals;
    Index total_elements = 1;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = num_basis_[i] + degree_[i];
        total_elements *= intervals[i];
    }

    if (elem_idx >= total_elements) {
        throw std::invalid_argument(
            "elem_idx is out of bounds for the element grid."
        );
    }

    // Decode the flat element index into per-direction span indices
    // using row-major (last index fastest) convention matching the assembler
    std::array<Index, d> span_indices;
    Index temp = elem_idx;
    for (std::size_t i = d; i-- > 0; ) {
        span_indices[i] = temp % intervals[i];
        temp /= intervals[i];
    }

    return get_element_dofs(span_indices);
}

template <std::size_t d>
std::vector<Index> DofMapper<d>::get_element_dofs(const std::array<Index, d>& span_indices) const
{
    // For each direction, collect the (degree+1) active basis function indices.
    // For span s in direction i the active basis functions are:
    //   max(0, s - degree_[i]) .. min(s, num_basis_[i] - 1)
    std::array<std::vector<Index>, d> active_1d;
    for (std::size_t i = 0; i < d; ++i) {
        Index s = span_indices[i];
        Index lo = (s >= degree_[i]) ? (s - degree_[i]) : 0;
        Index hi = std::min(s, num_basis_[i] - 1);
        for (Index k = lo; k <= hi; ++k) {
            active_1d[i].push_back(k);
        }
    }

    // Build the tensor-product of the per-direction 1D index sets and
    // map each multi-index to a global DOF
    std::vector<Index> dofs;
    std::array<Index, d> logical_idx;
    logical_idx.fill(0);

    // Recursive lambda to iterate over the tensor product of 1D index sets
    auto iterate = [&](auto& self, std::size_t dim) -> void {
        if (dim == d) {
            dofs.push_back(to_global(logical_idx));
            return;
        }
        for (Index k : active_1d[dim]) {
            logical_idx[dim] = k;
            self(self, dim + 1);
        }
    };

    iterate(iterate, 0);
    // NOTE: Do NOT sort — the order must match the tensor-product column
    // ordering (dim-0 outer, dim-1 inner, …) so that basis function
    // column k corresponds to control-point / DOF entry k.
    return dofs;
}

template <std::size_t d>
bool DofMapper<d>::next_logical_index(std::array<Index, d>& logical_idx) const
{
    for (std::size_t i = 0; i < d; ++i) {
        logical_idx[i]++;
        if (logical_idx[i] < num_basis_[i]) {
            return true;
        }
        logical_idx[i] = 0;
    }
    return false;
}

// === Template Instantiations ========================================================

template class DofMapper<1>;
template class DofMapper<2>;
template class DofMapper<3>;

} // namespace pyck
