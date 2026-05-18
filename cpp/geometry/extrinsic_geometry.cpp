#include "extrinsic_geometry.hpp"

#include <Eigen/Core>

namespace pyck
{

template <std::floating_point T, std::size_t d, std::size_t k>
    requires (d == 2)
ExtrinsicGeometry<T, d, k>::ExtrinsicGeometry(const IntrinsicGeometry<T, d, k>& ig)
{
    const Index Q = ig.a[0].rows();
    n.resize(Q, 3);

    // --- Normal -----------------------------------------------------------------

    // a_3 = (a_0 × a_1) / J; fallback to (0,0,1) at near-degenerate points.
    for (Index q = 0; q < Q; ++q)
    {
        Eigen::Matrix<T, 3, 1> v =
            ig.a[0].row(q).transpose().cross(ig.a[1].row(q).transpose());
        const T J = ig.jac(q);
        if (J > T(1e-14)) v /= J;
        else              v = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
        n.row(q) = v.transpose();
    }

    // --- Normal derivatives -----------------------------------------------------

    if constexpr (k >= 2)
    {
        for (std::size_t b = 0; b < 2; ++b)
            n_d1[b].resize(Q, 3);

        for (Index q = 0; q < Q; ++q)
        {
            const Eigen::Matrix<T, 3, 1> a0  = ig.a[0].row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a1  = ig.a[1].row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a3  = n.row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a00 = ig.a_d1[0][0].row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a01 = ig.a_d1[0][1].row(q).transpose();
            const Eigen::Matrix<T, 3, 1> a11 = ig.a_d1[1][1].row(q).transpose();

            const T inv_J = (ig.jac(q) > T(1e-14)) ? T(1) / ig.jac(q) : T(0);

            // ∂_β N where N = a_0 × a_1, β ∈ {0, 1}.
            const Eigen::Matrix<T, 3, 1> dN0 = a00.cross(a1) + a0.cross(a01);
            const Eigen::Matrix<T, 3, 1> dN1 = a01.cross(a1) + a0.cross(a11);

            // Tangent-plane projection: ∂_β a_3 = [(I - a_3 a_3ᵀ) ∂_β N] / J.
            n_d1[0].row(q) = ((dN0 - a3.dot(dN0) * a3) * inv_J).transpose();
            n_d1[1].row(q) = ((dN1 - a3.dot(dN1) * a3) * inv_J).transpose();
        }
    }
}

// === Surface Curvature ==============================================================

template <std::floating_point T, std::size_t d, std::size_t k>
    requires (d == 2)
void ExtrinsicGeometry<T, d, k>::compute_curvature(const IntrinsicGeometry<T, d, k>& ig)
{
    if constexpr (k >= 2)
    {
        const Index Q = ig.a[0].rows();

        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = i; j < 2; ++j)
                curv.b[i][j].resize(Q);
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j)
                curv.b_mixed[i][j].resize(Q);

        for (Index q = 0; q < Q; ++q)
        {
            // b_{αβ} = a_{αβ} · a_3   (sym, α ≤ β filled)
            const T b11 = ig.a_d1[0][0].row(q).dot(n.row(q));
            const T b12 = ig.a_d1[0][1].row(q).dot(n.row(q));
            const T b22 = ig.a_d1[1][1].row(q).dot(n.row(q));
            curv.b[0][0](q) = b11;
            curv.b[0][1](q) = b12;
            curv.b[1][1](q) = b22;

            // b^α_β = g^{αγ} b_{γβ}   (4 entries, not symmetric)
            const T gi11 = ig.g_inv[0][0](q);
            const T gi12 = ig.g_inv[0][1](q);
            const T gi22 = ig.g_inv[1][1](q);
            curv.b_mixed[0][0](q) = gi11 * b11 + gi12 * b12;
            curv.b_mixed[0][1](q) = gi11 * b12 + gi12 * b22;
            curv.b_mixed[1][0](q) = gi12 * b11 + gi22 * b12;
            curv.b_mixed[1][1](q) = gi12 * b12 + gi22 * b22;
        }
    }
}

// === Template Instantiations ========================================================

template class ExtrinsicGeometry<double, 2, 1>;
template class ExtrinsicGeometry<double, 2, 2>;
template class ExtrinsicGeometry<double, 2, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ExtrinsicGeometry<float, 2, 1>;
template class ExtrinsicGeometry<float, 2, 2>;
template class ExtrinsicGeometry<float, 2, 3>;
#endif

} // namespace pyck
