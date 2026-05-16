#include "basis_derivs.hpp"

#include <utility>

namespace pyck
{

template <std::floating_point T>
BasisDerivs<T, 1>
eval_basis(const Patch<T, 1>& patch,
           const ColMatrix<T, 1>& eval_coords,
           Index span_idx,
           std::size_t derivs_order)
{
    std::array<Index, 1> spans = {span_idx};
    auto raw = patch.tensor_product().eval_on_span(eval_coords, spans,
                                                   static_cast<Index>(derivs_order));

    BasisDerivs<T, 1> b;
    b.order = static_cast<Index>(derivs_order);
    b.N = std::move(raw[0]);

    if (b.order >= 1) b.N_u   = std::move(raw[1]);
    if (b.order >= 2) b.N_uu  = std::move(raw[2]);
    if (b.order >= 3) b.N_uuu = std::move(raw[3]);
    return b;
}

template <std::floating_point T>
BasisDerivs<T, 2>
eval_basis(const Patch<T, 2>& patch,
           const ColMatrix<T, 2>& eval_coords,
           Index span_idx,
           std::size_t derivs_order)
{
    auto spans = patch.decode_span(span_idx);
    auto raw = patch.tensor_product().eval_on_span(eval_coords, spans,
                                                   static_cast<Index>(derivs_order));

    BasisDerivs<T, 2> b;
    b.order = static_cast<Index>(derivs_order);
    const Index S = b.order + 1;
    b.N = std::move(raw[0]);

    if (b.order >= 1)
    {
        b.N_u = std::move(raw[1 * S + 0]);
        b.N_v = std::move(raw[0 * S + 1]);
    }
    if (b.order >= 2)
    {
        b.N_uu = std::move(raw[2 * S + 0]);
        b.N_uv = std::move(raw[1 * S + 1]);
        b.N_vv = std::move(raw[0 * S + 2]);
    }
    if (b.order >= 3)
    {
        b.N_uuu = std::move(raw[3 * S + 0]);
        b.N_uuv = std::move(raw[2 * S + 1]);
        b.N_uvv = std::move(raw[1 * S + 2]);
        b.N_vvv = std::move(raw[0 * S + 3]);
    }
    return b;
}

template <std::floating_point T>
BasisDerivs<T, 3>
eval_basis(const Patch<T, 3>& patch,
           const ColMatrix<T, 3>& eval_coords,
           Index span_idx,
           std::size_t derivs_order)
{
    auto spans = patch.decode_span(span_idx);
    auto raw = patch.tensor_product().eval_on_span(eval_coords, spans,
                                                   static_cast<Index>(derivs_order));

    BasisDerivs<T, 3> b;
    b.order = static_cast<Index>(derivs_order);
    const Index S  = b.order + 1;
    const Index S2 = S * S;
    auto at = [&](Index a, Index bb, Index c) -> Index { return a * S2 + bb * S + c; };

    b.N = std::move(raw[at(0, 0, 0)]);

    if (b.order >= 1)
    {
        b.N_u = std::move(raw[at(1, 0, 0)]);
        b.N_v = std::move(raw[at(0, 1, 0)]);
        b.N_w = std::move(raw[at(0, 0, 1)]);
    }
    if (b.order >= 2)
    {
        b.N_uu = std::move(raw[at(2, 0, 0)]);
        b.N_uv = std::move(raw[at(1, 1, 0)]);
        b.N_uw = std::move(raw[at(1, 0, 1)]);
        b.N_vv = std::move(raw[at(0, 2, 0)]);
        b.N_vw = std::move(raw[at(0, 1, 1)]);
        b.N_ww = std::move(raw[at(0, 0, 2)]);
    }
    if (b.order >= 3)
    {
        b.N_uuu = std::move(raw[at(3, 0, 0)]);
        b.N_uuv = std::move(raw[at(2, 1, 0)]);
        b.N_uuw = std::move(raw[at(2, 0, 1)]);
        b.N_uvv = std::move(raw[at(1, 2, 0)]);
        b.N_uvw = std::move(raw[at(1, 1, 1)]);
        b.N_uww = std::move(raw[at(1, 0, 2)]);
        b.N_vvv = std::move(raw[at(0, 3, 0)]);
        b.N_vvw = std::move(raw[at(0, 2, 1)]);
        b.N_vww = std::move(raw[at(0, 1, 2)]);
        b.N_www = std::move(raw[at(0, 0, 3)]);
    }
    return b;
}

// === Template Specializations =======================================================

template BasisDerivs<double, 1> eval_basis<double>(
    const Patch<double, 1>&, const ColMatrix<double, 1>&, Index, std::size_t);
template BasisDerivs<double, 2> eval_basis<double>(
    const Patch<double, 2>&, const ColMatrix<double, 2>&, Index, std::size_t);
template BasisDerivs<double, 3> eval_basis<double>(
    const Patch<double, 3>&, const ColMatrix<double, 3>&, Index, std::size_t);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template BasisDerivs<float, 1> eval_basis<float>(
    const Patch<float, 1>&, const ColMatrix<float, 1>&, Index, std::size_t);
template BasisDerivs<float, 2> eval_basis<float>(
    const Patch<float, 2>&, const ColMatrix<float, 2>&, Index, std::size_t);
template BasisDerivs<float, 3> eval_basis<float>(
    const Patch<float, 3>&, const ColMatrix<float, 3>&, Index, std::size_t);
#endif

} // namespace pyck
