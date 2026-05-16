#include "local_frame.hpp"

#include <cmath>

namespace pyck
{

template <std::floating_point T>
LocalFrame<T, 1>
eval_local_frame(const BasisDerivs<T, 1>& basis,
                 const ColMatrix<T, 3>& act_pts)
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

template <std::floating_point T>
LocalFrame<T, 2>
eval_local_frame(const BasisDerivs<T, 2>& basis,
                 const ColMatrix<T, 3>& act_pts)
{
    const Index Q = basis.N.rows();

    LocalFrame<T, 2> lf;

    // Tangents a_α (order ≥ 1).
    lf.a1 = basis.N_u * act_pts;
    lf.a2 = basis.N_v * act_pts;

    // Tangent derivatives a_{αβ} (order ≥ 2).
    if (basis.order >= 2) {
        lf.a11 = basis.N_uu * act_pts;
        lf.a12 = basis.N_uv * act_pts;
        lf.a22 = basis.N_vv * act_pts;
    }

    // 2nd tangent derivatives a_{αβγ} (order ≥ 3).
    if (basis.order >= 3) {
        lf.a111 = basis.N_uuu * act_pts;
        lf.a112 = basis.N_uuv * act_pts;
        lf.a122 = basis.N_uvv * act_pts;
        lf.a222 = basis.N_vvv * act_pts;
    }

    lf.g.resize(Q, 3);
    lf.g_inv.resize(Q, 3);
    lf.jac.resize(Q);

    for (Index q = 0; q < Q; ++q) {
        const auto a1_q = lf.a1.row(q);
        const auto a2_q = lf.a2.row(q);
        const T g11 = a1_q.squaredNorm();
        const T g12 = a1_q.dot(a2_q);
        const T g22 = a2_q.squaredNorm();
        const T det_g = g11 * g22 - g12 * g12;
        const T inv_dg = T(1) / det_g;
        lf.g(q, 0) = g11;
        lf.g(q, 1) = g12;
        lf.g(q, 2) = g22;
        lf.g_inv(q, 0) =  g22 * inv_dg;
        lf.g_inv(q, 1) = -g12 * inv_dg;
        lf.g_inv(q, 2) =  g11 * inv_dg;
        lf.jac(q) = std::sqrt(det_g);
    }
    return lf;
}

template <std::floating_point T>
LocalFrame<T, 3>
eval_local_frame(const BasisDerivs<T, 3>& basis,
                 const ColMatrix<T, 3>& act_pts)
{
    const Index Q = basis.N.rows();

    LocalFrame<T, 3> lf;

    // Tangents a_α (order ≥ 1).
    lf.a1 = basis.N_u * act_pts;
    lf.a2 = basis.N_v * act_pts;
    lf.a3 = basis.N_w * act_pts;

    // Tangent derivatives a_{αβ} (order ≥ 2).
    if (basis.order >= 2) {
        lf.a11 = basis.N_uu * act_pts;
        lf.a12 = basis.N_uv * act_pts;
        lf.a13 = basis.N_uw * act_pts;
        lf.a22 = basis.N_vv * act_pts;
        lf.a23 = basis.N_vw * act_pts;
        lf.a33 = basis.N_ww * act_pts;
    }

    // 2nd tangent derivatives a_{αβγ} (order ≥ 3).
    if (basis.order >= 3) {
        lf.a111 = basis.N_uuu * act_pts;
        lf.a112 = basis.N_uuv * act_pts;
        lf.a113 = basis.N_uuw * act_pts;
        lf.a122 = basis.N_uvv * act_pts;
        lf.a123 = basis.N_uvw * act_pts;
        lf.a133 = basis.N_uww * act_pts;
        lf.a222 = basis.N_vvv * act_pts;
        lf.a223 = basis.N_vvw * act_pts;
        lf.a233 = basis.N_vww * act_pts;
        lf.a333 = basis.N_www * act_pts;
    }

    lf.g.resize(Q, 6);
    lf.g_inv.resize(Q, 6);
    lf.jac.resize(Q);

    for (Index q = 0; q < Q; ++q) {
        const auto a1_q = lf.a1.row(q);
        const auto a2_q = lf.a2.row(q);
        const auto a3_q = lf.a3.row(q);

        const T g11 = a1_q.squaredNorm();
        const T g12 = a1_q.dot(a2_q);
        const T g13 = a1_q.dot(a3_q);
        const T g22 = a2_q.squaredNorm();
        const T g23 = a2_q.dot(a3_q);
        const T g33 = a3_q.squaredNorm();

        // Symmetric 3×3 cofactors: C_ij of the metric.
        const T c11 = g22 * g33 - g23 * g23;
        const T c12 = g23 * g13 - g12 * g33;
        const T c13 = g12 * g23 - g22 * g13;
        const T c22 = g11 * g33 - g13 * g13;
        const T c23 = g13 * g12 - g11 * g23;
        const T c33 = g11 * g22 - g12 * g12;
        const T det_g = g11 * c11 + g12 * c12 + g13 * c13;
        const T inv_dg = T(1) / det_g;

        lf.g(q, 0) = g11; lf.g(q, 1) = g12; lf.g(q, 2) = g13;
        lf.g(q, 3) = g22; lf.g(q, 4) = g23; lf.g(q, 5) = g33;

        lf.g_inv(q, 0) = c11 * inv_dg;
        lf.g_inv(q, 1) = c12 * inv_dg;
        lf.g_inv(q, 2) = c13 * inv_dg;
        lf.g_inv(q, 3) = c22 * inv_dg;
        lf.g_inv(q, 4) = c23 * inv_dg;
        lf.g_inv(q, 5) = c33 * inv_dg;

        lf.jac(q) = std::sqrt(det_g);
    }
    return lf;
}

// === Template Specializations =======================================================

template LocalFrame<double, 1> eval_local_frame<double>(
    const BasisDerivs<double, 1>&, const ColMatrix<double, 3>&);
template LocalFrame<double, 2> eval_local_frame<double>(
    const BasisDerivs<double, 2>&, const ColMatrix<double, 3>&);
template LocalFrame<double, 3> eval_local_frame<double>(
    const BasisDerivs<double, 3>&, const ColMatrix<double, 3>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template LocalFrame<float, 1> eval_local_frame<float>(
    const BasisDerivs<float, 1>&, const ColMatrix<float, 3>&);
template LocalFrame<float, 2> eval_local_frame<float>(
    const BasisDerivs<float, 2>&, const ColMatrix<float, 3>&);
template LocalFrame<float, 3> eval_local_frame<float>(
    const BasisDerivs<float, 3>&, const ColMatrix<float, 3>&);
#endif

} // namespace pyck
