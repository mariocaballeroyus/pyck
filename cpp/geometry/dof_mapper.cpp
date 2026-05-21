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
std::vector<Index> DofMapper<d>::get_layer_dofs(Index param_dim,
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
void DofMapper<d>::get_element_dofs(Index elem_idx, std::vector<Index>& out) const
{
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

    std::array<Index, d> span_indices;
    Index temp = elem_idx;
    for (std::size_t i = 0; i < d; ++i) {
        span_indices[i] = temp % intervals[i];
        temp /= intervals[i];
    }

    get_element_dofs(span_indices, out);
}

template <std::size_t d>
void DofMapper<d>::get_element_dofs(const std::array<Index, d>& span_indices,
                                    std::vector<Index>& out) const
{
    // Active 1D indices in direction i are [lo_i, hi_i] with
    //   lo_i = max(0, s_i - p_i),  hi_i = min(s_i, n_i - 1).
    // Output column order: dim-0 outer, dim-(d-1) inner, matching
    // TensorProduct::eval_on_span's tensor-product column layout.
    if constexpr (d == 1) {
        const Index p = degree_[0];
        const Index n = num_basis_[0];
        const Index s = span_indices[0];
        const Index lo = (s >= p) ? s - p : 0;
        const Index hi = std::min(s, n - 1);
        const Index N  = hi - lo + 1;

        out.resize(static_cast<std::size_t>(N));
        for (Index k = 0; k < N; ++k) out[k] = lo + k;
    }
    else if constexpr (d == 2) {
        const Index p_u = degree_[0],    p_v = degree_[1];
        const Index n_u = num_basis_[0], n_v = num_basis_[1];
        const Index s_u = span_indices[0], s_v = span_indices[1];

        const Index lo_u = (s_u >= p_u) ? s_u - p_u : 0;
        const Index hi_u = std::min(s_u, n_u - 1);
        const Index lo_v = (s_v >= p_v) ? s_v - p_v : 0;
        const Index hi_v = std::min(s_v, n_v - 1);

        const Index N_u = hi_u - lo_u + 1;
        const Index N_v = hi_v - lo_v + 1;

        out.resize(static_cast<std::size_t>(N_u * N_v));
        Index w = 0;
        for (Index iu = 0; iu < N_u; ++iu) {
            const Index cp_u = lo_u + iu;
            for (Index iv = 0; iv < N_v; ++iv) {
                out[w++] = cp_u + (lo_v + iv) * n_u;
            }
        }
    }
    else if constexpr (d == 3) {
        const Index p_u = degree_[0],    p_v = degree_[1],    p_w = degree_[2];
        const Index n_u = num_basis_[0], n_v = num_basis_[1], n_w = num_basis_[2];
        const Index s_u = span_indices[0], s_v = span_indices[1], s_w = span_indices[2];

        const Index lo_u = (s_u >= p_u) ? s_u - p_u : 0;
        const Index hi_u = std::min(s_u, n_u - 1);
        const Index lo_v = (s_v >= p_v) ? s_v - p_v : 0;
        const Index hi_v = std::min(s_v, n_v - 1);
        const Index lo_w = (s_w >= p_w) ? s_w - p_w : 0;
        const Index hi_w = std::min(s_w, n_w - 1);

        const Index N_u = hi_u - lo_u + 1;
        const Index N_v = hi_v - lo_v + 1;
        const Index N_w = hi_w - lo_w + 1;

        out.resize(static_cast<std::size_t>(N_u * N_v * N_w));
        const Index stride_w = n_u * n_v;
        Index w_ = 0;
        for (Index iu = 0; iu < N_u; ++iu) {
            const Index cp_u = lo_u + iu;
            for (Index iv = 0; iv < N_v; ++iv) {
                const Index cp_uv = cp_u + (lo_v + iv) * n_u;
                for (Index iw = 0; iw < N_w; ++iw) {
                    out[w_++] = cp_uv + (lo_w + iw) * stride_w;
                }
            }
        }
    }
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
