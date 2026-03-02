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
