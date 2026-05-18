#include "extrinsic_geometry.hpp"

#include <Eigen/Core>

namespace pyck
{

template <std::floating_point T, std::size_t d>
    requires (d == 2)
ExtrinsicGeometry<T, d>::ExtrinsicGeometry(const IntrinsicGeometry<T, d>& ig)
{
    const Index Q_ = ig.Q();
    n.resize(Q_, 3);

    // --- Normal -----------------------------------------------------------------

    // a_3 = (a_0 × a_1) / J; fallback to (0,0,1) at near-degenerate points.
    for (Index q = 0; q < Q_; ++q)
    {
        Eigen::Matrix<T, 3, 1> v =
            ig.a(0).row(q).transpose().cross(ig.a(1).row(q).transpose());
        const T J = ig.jac(q);
        if (J > T(1e-14)) v /= J;
        else              v = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
        n.row(q) = v.transpose();
    }

    // --- Normal derivatives -----------------------------------------------------

    if (ig.order() >= 2)
    {
        n_d1_data.resize(Q_ * 2, 3);

        for (Index q = 0; q < Q_; ++q)
        {
            const Eigen::Matrix<T, 3, 1> a0  = ig.a(0).row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a1  = ig.a(1).row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a3  = n.row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a00 = ig.a_d1(0, 0).row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a01 = ig.a_d1(0, 1).row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a11 = ig.a_d1(1, 1).row(q).transpose();

            const T inv_J = (ig.jac(q) > T(1e-14)) ? T(1) / ig.jac(q) : T(0);

            // ∂_β N where N = a_0 × a_1, β ∈ {0, 1}.
            const Eigen::Matrix<T, 3, 1> dN0 = a00.cross(a1) + a0.cross(a01);
            const Eigen::Matrix<T, 3, 1> dN1 = a01.cross(a1) + a0.cross(a11);

            // Tangent-plane projection: ∂_β a_3 = [(I - a_3 a_3ᵀ) ∂_β N] / J.
            n_d1_data.middleRows(0 * Q_, Q_).row(q) =
                ((dN0 - a3.dot(dN0) * a3) * inv_J).transpose();
            n_d1_data.middleRows(1 * Q_, Q_).row(q) =
                ((dN1 - a3.dot(dN1) * a3) * inv_J).transpose();
        }
    }
}

// === Surface Curvature ==============================================================

template <std::floating_point T, std::size_t d>
    requires (d == 2)
void ExtrinsicGeometry<T, d>::compute_curvature(const IntrinsicGeometry<T, d>& ig)
{
    if (ig.order() < 2) return;

    const Index Q_ = ig.Q();
    curv.b_data.resize(Q_, 3);          // n_metric = 3 for d=2
    curv.b_mixed_data.resize(Q_, 4);    // 2 × 2 full

    for (Index q = 0; q < Q_; ++q)
    {
        // b_{αβ} = a_{αβ} · a_3   (sym, α ≤ β filled via pack2)
        const T b11 = ig.a_d1(0, 0).row(q).dot(n.row(q));
        const T b12 = ig.a_d1(0, 1).row(q).dot(n.row(q));
        const T b22 = ig.a_d1(1, 1).row(q).dot(n.row(q));
        curv.b_data(q, pack2<2>(0, 0)) = b11;
        curv.b_data(q, pack2<2>(0, 1)) = b12;
        curv.b_data(q, pack2<2>(1, 1)) = b22;

        // b^α_β = g^{αγ} b_{γβ}   (4 entries, not symmetric)
        const T gi11 = ig.g_inv(0, 0)(q);
        const T gi12 = ig.g_inv(0, 1)(q);
        const T gi22 = ig.g_inv(1, 1)(q);
        curv.b_mixed_data(q, 0 * 2 + 0) = gi11 * b11 + gi12 * b12;
        curv.b_mixed_data(q, 0 * 2 + 1) = gi11 * b12 + gi12 * b22;
        curv.b_mixed_data(q, 1 * 2 + 0) = gi12 * b11 + gi22 * b12;
        curv.b_mixed_data(q, 1 * 2 + 1) = gi12 * b12 + gi22 * b22;
    }
}

// === Template Instantiations ========================================================

template class ExtrinsicGeometry<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ExtrinsicGeometry<float, 2>;
#endif

} // namespace pyck
