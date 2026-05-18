#include "basis_derivs.hpp"

#include <utility>

namespace pyck
{

namespace {

/// Linearise a multi-index of partial-derivative orders into the flat index of
/// the `raw[]` buffer returned by `tensor_product::eval_on_span`. The buffer
/// layout is dim-0-outermost, dim-(d-1)-innermost; with S = order + 1, the
/// flat index of `(i_0, i_1, …, i_{d-1})` is Σ i_k · S^{d-1-k}.
template <std::size_t d>
Index raw_flat(const std::array<Index, d>& derivs, Index S)
{
    Index idx = 0;
    for (std::size_t k = 0; k < d; ++k) {
        idx = idx * S + derivs[k];
    }
    return idx;
}

}  // namespace

template <std::floating_point T, std::size_t d>
BasisDerivs<T, d>
eval_basis(const Patch<T, d>& patch,
           const std::type_identity_t<ColMatrix<T, d>>& eval_coords,
           Index span_idx,
           std::size_t derivs_order)
{
    const Index order = static_cast<Index>(derivs_order);
    const Index S     = order + 1;
    const auto spans  = patch.decode_span(span_idx);

    std::vector<Matrix<T>> raw;
    patch.tensor_product().eval_on_span(eval_coords, spans, order, raw);

    BasisDerivs<T, d> b;
    b.order = order;

    // 0th derivative — values N (multi-index all-zero).
    {
        std::array<Index, d> derivs{};
        b.N = std::move(raw[raw_flat<d>(derivs, S)]);
    }

    if (order >= 1) 
    {
        for (std::size_t i = 0; i < d; ++i) 
        {
            std::array<Index, d> derivs{};
            derivs[i] = 1;
            b.N_d1[i] = std::move(raw[raw_flat<d>(derivs, S)]);
        }
    }

    if (order >= 2) 
    {
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j) 
            {
                std::array<Index, d> derivs{};
                derivs[i] += 1; derivs[j] += 1;
                b.N_d2[i][j] = std::move(raw[raw_flat<d>(derivs, S)]);
            }
    }

    if (order >= 3) 
    {
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j)
                for (std::size_t k = j; k < d; ++k) 
                {
                    std::array<Index, d> derivs{};
                    derivs[i] += 1; derivs[j] += 1; derivs[k] += 1;
                    b.N_d3[i][j][k] = std::move(raw[raw_flat<d>(derivs, S)]);
                }
    }

    return b;
}

// === Template Instantiations ========================================================

template BasisDerivs<double, 1> eval_basis<double, 1>(
    const Patch<double, 1>&, const ColMatrix<double, 1>&, Index, std::size_t);
template BasisDerivs<double, 2> eval_basis<double, 2>(
    const Patch<double, 2>&, const ColMatrix<double, 2>&, Index, std::size_t);
template BasisDerivs<double, 3> eval_basis<double, 3>(
    const Patch<double, 3>&, const ColMatrix<double, 3>&, Index, std::size_t);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template BasisDerivs<float, 1> eval_basis<float, 1>(
    const Patch<float, 1>&, const ColMatrix<float, 1>&, Index, std::size_t);
template BasisDerivs<float, 2> eval_basis<float, 2>(
    const Patch<float, 2>&, const ColMatrix<float, 2>&, Index, std::size_t);
template BasisDerivs<float, 3> eval_basis<float, 3>(
    const Patch<float, 3>&, const ColMatrix<float, 3>&, Index, std::size_t);
#endif

} // namespace pyck
