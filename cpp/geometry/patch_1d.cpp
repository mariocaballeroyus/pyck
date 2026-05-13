#include "patch.hpp"
#include "bspline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "../quadrature/quadrature.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u, const ColMatrix<T, 3>& control_pts) requires(d == 1)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u)),
      dof_mapper_({tensor_product_.basis(0).num_basis()}, {tensor_product_.basis(0).degree()})
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

    // Initialize Greville points and spans
    Vector<T> gu = this->tensor_product_.basis(0).greville_abscissae();
    greville_points_.resize(gu.size(), 1);
    greville_spans_.resize(gu.size());
    for (Index i = 0; i < static_cast<Index>(gu.size()); ++i) {
        greville_points_(i, 0) = gu[i];
        greville_spans_[i] = this->tensor_product_.basis(0).find_span(gu[i]);
    }
}

// ============================================================================
// Composable geometric primitives — 1D
// ============================================================================

template <std::floating_point T, std::size_t d>
BasisDerivs<T, d>
Patch<T, d>::eval_basis(const ColMatrix<T, d>& eval_coords,
                        Index span_idx,
                        std::size_t derivs_order) const requires(d == 1)
{
    std::array<Index, 1> spans = {span_idx};
    auto raw = this->tensor_product().eval_on_span(eval_coords, spans,
                                                   static_cast<Index>(derivs_order));
    BasisDerivs<T, 1> b;
    b.order = static_cast<Index>(derivs_order);
    b.N = std::move(raw[0]);
    if (b.order >= 1) b.N_u   = std::move(raw[1]);
    if (b.order >= 2) b.N_uu  = std::move(raw[2]);
    if (b.order >= 3) b.N_uuu = std::move(raw[3]);
    return b;
}

template <std::floating_point T, std::size_t d>
LocalFrame<T, d>
Patch<T, d>::eval_local_frame(const BasisDerivs<T, d>& basis,
                              const ColMatrix<T, 3>& act_pts) const requires(d == 1)
{
    const Index Q = basis.N.rows();

    LocalFrame<T, 1> lf;
    lf.a1 = basis.N_u * act_pts;
    if (basis.order >= 2) lf.a11  = basis.N_uu  * act_pts;
    if (basis.order >= 3) lf.a111 = basis.N_uuu * act_pts;

    lf.g11.resize(Q);
    lf.g_inv_11.resize(Q);
    lf.jac.resize(Q);
    for (Index q = 0; q < Q; ++q) {
        const T g11 = lf.a1.row(q).squaredNorm();
        lf.g11(q) = g11;
        lf.g_inv_11(q) = T(1) / g11;
        lf.jac(q) = std::sqrt(g11);
    }
    return lf;
}

template <std::floating_point T, std::size_t d>
ChristoffelSymbols<T, d>
Patch<T, d>::eval_christoffel(const LocalFrame<T, d>& local) const requires(d == 1)
{
    const Index Q = local.a1.rows();
    ChristoffelSymbols<T, d> chr;
    chr.G1_11.resize(Q);
    for (Index q = 0; q < Q; ++q) {
        // Γ¹_{11} = (a_1 · a_{1,1}) / g_{11}
        chr.G1_11(q) = local.a1.row(q).dot(local.a11.row(q)) * local.g_inv_11(q);
    }
    return chr;
}

template class Patch<double, 1>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 1>;
#endif

} // namespace pyck
