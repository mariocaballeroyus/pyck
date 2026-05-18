#include "intrinsic_geometry.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace pyck
{

template <std::floating_point T, std::size_t d, std::size_t k>
IntrinsicGeometry<T, d, k>::IntrinsicGeometry(const BasisDerivs<T, d>& basis,
                                              const ColMatrix<T, 3>& act_pts)
{
    // --- Order 1 --------------------------------------------------------------------

    x = basis.N * act_pts;
    for (std::size_t i = 0; i < d; ++i) {
        a[i] = basis.N_d1[i] * act_pts;
    }

    // --- Metric and Jacobian --------------------------------------------------------

    const Index Q = a[0].rows();

    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = i; j < d; ++j) {
            g    [i][j].resize(Q);
            g_inv[i][j].resize(Q);
        }
    }
    jac.resize(Q);

    for (Index q = 0; q < Q; ++q)
    {
        Eigen::Matrix<T, d, d> g_mat;
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = i; j < d; ++j) {
                const T g_ij = a[i].row(q).dot(a[j].row(q));
                g_mat(i, j) = g_ij;
                if (j != i) g_mat(j, i) = g_ij;
            }
        }

        const Eigen::Matrix<T, d, d> inv_g = g_mat.inverse();
        const T det_g = g_mat.determinant();

        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = i; j < d; ++j) {
                g    [i][j](q) = g_mat(i, j);
                g_inv[i][j](q) = inv_g(i, j);
            }
        }
        jac(q) = std::sqrt(det_g);
    }

    // --- Order 2 --------------------------------------------------------------------
    if constexpr (k >= 2)
    {
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j) {
                if (basis.N_d2[i][j].size() > 0)
                    a_d1[i][j] = basis.N_d2[i][j] * act_pts;
                else
                    a_d1[i][j].setZero(Q, 3);
            }
    }

    // --- Order 3 --------------------------------------------------------------------

    if constexpr (k >= 3)
    {
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j)
                for (std::size_t idx = j; idx < d; ++idx) {
                    if (basis.N_d3[i][j][idx].size() > 0)
                        a_d2[i][j][idx] = basis.N_d3[i][j][idx] * act_pts;
                    else
                        a_d2[i][j][idx].setZero(Q, 3);
                }
    }
}

// === Christoffel Symbols of the Second Kind =========================================

template <std::floating_point T, std::size_t d, std::size_t k>
void IntrinsicGeometry<T, d, k>::compute_christoffels()
{
    const Index Q = a[0].rows();

    if constexpr (k >= 2)
    {
        for (std::size_t m = 0; m < d; ++m)
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = i; j < d; ++j)
                    chr.Gamma[m][i][j].resize(Q);
    }

    if constexpr (k >= 3)
    {
        for (std::size_t m = 0; m < d; ++m)
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = i; j < d; ++j)
                    for (std::size_t g = 0; g < d; ++g)
                        chr.Gamma_d1[m][i][j][g].resize(Q);
    }

    for (Index q = 0; q < Q; ++q)
    {
        // --- Order 1 ----------------------------------------------------------------

        // Densify g^{αβ} from upper-triangular storage.
        Eigen::Matrix<T, d, d> g_inv_full;
        for (std::size_t alpha = 0; alpha < d; ++alpha) {
            for (std::size_t beta = alpha; beta < d; ++beta) {
                g_inv_full(alpha, beta) = g_inv[alpha][beta](q);
                if (beta != alpha) g_inv_full(beta, alpha) = g_inv_full(alpha, beta);
            }
        }

        // a^m = g^{ml} a_l (contravariant tangents).
        std::array<Eigen::RowVector<T, 3>, d> aup;
        for (std::size_t m = 0; m < d; ++m)
        {
            aup[m].setZero();
            for (std::size_t l = 0; l < d; ++l)
                aup[m] += g_inv_full(m, l) * a[l].row(q);
        }

        // --- Order 2 ----------------------------------------------------------------

        if constexpr (k >= 2)
        {
            // Γ^m_{αβ} = a^m · a_{αβ}.
            for (std::size_t m = 0; m < d; ++m)
                for (std::size_t i = 0; i < d; ++i)
                    for (std::size_t j = i; j < d; ++j)
                        chr.Gamma[m][i][j](q) = aup[m].dot(a_d1[i][j].row(q));
        }

        // --- Order 3 ----------------------------------------------------------------

        if constexpr (k >= 3)
        {
            // ∂_w g_{uv}: stored densely in dg[w], symmetric in (u,v).
            std::array<Eigen::Matrix<T, d, d>, d> dg;
            for (std::size_t w = 0; w < d; ++w) {
                for (std::size_t u = 0; u < d; ++u) {
                    for (std::size_t v = u; v < d; ++v) 
                    {
                        const std::size_t uw0 = std::min(u, w), uw1 = std::max(u, w);
                        const std::size_t vw0 = std::min(v, w), vw1 = std::max(v, w);
                        const T val = a_d1[uw0][uw1].row(q).dot(a[v].row(q))
                                    + a[u].row(q).dot(a_d1[vw0][vw1].row(q));
                        dg[w](u, v) = val;
                        if (v != u) dg[w](v, u) = val;
                    }
                }
            }

            // ∂_γ g^{αβ} = -g^{-1} (∂_γ g) g^{-1}.
            std::array<Eigen::Matrix<T, d, d>, d> dginv;
            for (std::size_t gam = 0; gam < d; ++gam)
                dginv[gam].noalias() = -g_inv_full * dg[gam] * g_inv_full;

            // ∂_γ a^m = (∂_γ g^{ml}) a_l + g^{ml} a_{l,γ}.
            std::array<std::array<Eigen::RowVector<T, 3>, d>, d> daup;
            for (std::size_t m = 0; m < d; ++m) 
            {
                for (std::size_t gam = 0; gam < d; ++gam) 
                {
                    daup[m][gam].setZero();
                    for (std::size_t l = 0; l < d; ++l) 
                    {
                        daup[m][gam] += dginv[gam](m, l) * a[l].row(q);
                        const std::size_t lg0 = std::min(l, gam), lg1 = std::max(l, gam);
                        daup[m][gam] += g_inv_full(m, l) * a_d1[lg0][lg1].row(q);
                    }
                }
            }

            // ∂_γ Γ^m_{αβ} = (∂_γ a^m) · a_{αβ} + a^m · a_{αβγ}.
            // a_{αβγ} is stored at fully-sorted indices in a_d2.
            for (std::size_t m = 0; m < d; ++m)
                for (std::size_t i = 0; i < d; ++i)
                    for (std::size_t j = i; j < d; ++j)
                        for (std::size_t gam = 0; gam < d; ++gam) 
                        {
                            std::size_t s0 = i, s1 = j, s2 = gam;

                            if (s0 > s1) std::swap(s0, s1);
                            if (s1 > s2) std::swap(s1, s2);
                            if (s0 > s1) std::swap(s0, s1);

                            chr.Gamma_d1[m][i][j][gam](q) =
                                daup[m][gam].dot(a_d1[i][j].row(q))
                              + aup[m].dot(a_d2[s0][s1][s2].row(q));
                        }
        }
    }
}

// === Template Instantiations ========================================================

template class IntrinsicGeometry<double, 1, 1>;
template class IntrinsicGeometry<double, 1, 2>;
template class IntrinsicGeometry<double, 1, 3>;
template class IntrinsicGeometry<double, 2, 1>;
template class IntrinsicGeometry<double, 2, 2>;
template class IntrinsicGeometry<double, 2, 3>;
template class IntrinsicGeometry<double, 3, 1>;
template class IntrinsicGeometry<double, 3, 2>;
template class IntrinsicGeometry<double, 3, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class IntrinsicGeometry<float, 1, 1>;
template class IntrinsicGeometry<float, 1, 2>;
template class IntrinsicGeometry<float, 1, 3>;
template class IntrinsicGeometry<float, 2, 1>;
template class IntrinsicGeometry<float, 2, 2>;
template class IntrinsicGeometry<float, 2, 3>;
template class IntrinsicGeometry<float, 3, 1>;
template class IntrinsicGeometry<float, 3, 2>;
template class IntrinsicGeometry<float, 3, 3>;
#endif

} // namespace pyck
