#include "intrinsic_geometry.hpp"

#include <Eigen/Core>
#include <cmath>

namespace pyck
{

template <std::floating_point T, std::size_t d>
IntrinsicGeometry<T, d>::IntrinsicGeometry(const BasisDerivs<T, d>& basis,
                                           const ColMatrix<T, 3>& act_pts)
{
    const Index order_ = basis.order();
    const Index Q_     = basis.Q();
    constexpr Index n_metric = d * (d + 1) / 2;

    position_data.resize(order_ + 1);

    // --- Order 0: position ----------------------------------------------------------

    position_data[0].resize(Q_, 3);
    position_data[0].noalias() = basis.N() * act_pts;

    // --- Order 1: covariant basis ---------------------------------------------------

    if (order_ >= 1)
    {
        constexpr Index n_k = d;            // C(1+d-1, d-1) = d
        position_data[1].resize(Q_ * n_k, 3);
        for (Index dir = 0; dir < d; ++dir) {
            position_data[1].middleRows(dir * Q_, Q_).noalias() =
                basis.N_d1(dir) * act_pts;
        }
    }

    // --- Metric and Jacobian --------------------------------------------------------

    g_data.resize(Q_, n_metric);
    g_inv_data.resize(Q_, n_metric);
    jac.resize(Q_);

    for (Index q = 0; q < Q_; ++q)
    {
        Eigen::Matrix<T, d, d> g_mat;
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = i; j < d; ++j) {
                const T g_ij = a(i).row(q).dot(a(j).row(q));
                g_mat(i, j) = g_ij;
                if (j != i) g_mat(j, i) = g_ij;
            }
        }

        const Eigen::Matrix<T, d, d> inv_g = g_mat.inverse();
        const T det_g = g_mat.determinant();

        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = i; j < d; ++j) {
                const Index packed = pack2<d>(i, j);
                g_data    (q, packed) = g_mat(i, j);
                g_inv_data(q, packed) = inv_g(i, j);
            }
        }
        jac(q) = std::sqrt(det_g);
    }

    // --- Order 2: a_d1 = ∂_{αβ} x ---------------------------------------------------

    if (order_ >= 2)
    {
        constexpr Index n_k = d * (d + 1) / 2;
        position_data[2].resize(Q_ * n_k, 3);
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j) {
                const Index packed = pack2<d>(i, j);
                position_data[2].middleRows(packed * Q_, Q_).noalias() =
                    basis.N_d2(i, j) * act_pts;
            }
    }

    // --- Order 3: a_d2 = ∂_{αβγ} x --------------------------------------------------

    if (order_ >= 3)
    {
        const auto dirs_list = enumerate_direction_indices<d>(3);
        const Index n_k = static_cast<Index>(dirs_list.size());
        position_data[3].resize(Q_ * n_k, 3);
        for (Index packed = 0; packed < n_k; ++packed) {
            const auto& dirs = dirs_list[packed];
            position_data[3].middleRows(packed * Q_, Q_).noalias() =
                basis.N_d3(dirs[0], dirs[1], dirs[2]) * act_pts;
        }
    }
}

// === Christoffel Symbols of the Second Kind =========================================

template <std::floating_point T, std::size_t d>
void IntrinsicGeometry<T, d>::compute_christoffels()
{
    const Index ord = order();
    if (ord < 2) return;

    const Index Q_ = Q();
    constexpr Index n_d2 = d * (d + 1) / 2;

    chr.Gamma_data.resize(Q_, d * n_d2);
    if (ord >= 3) {
        chr.Gamma_d1_data.resize(Q_, d * n_d2 * d);
    }

    for (Index q = 0; q < Q_; ++q)
    {
        // Densify g^{αβ} from upper-tri storage.
        Eigen::Matrix<T, d, d> g_inv_full;
        for (std::size_t alpha = 0; alpha < d; ++alpha) {
            for (std::size_t beta = alpha; beta < d; ++beta) {
                g_inv_full(alpha, beta) = g_inv(alpha, beta)(q);
                if (beta != alpha) g_inv_full(beta, alpha) = g_inv_full(alpha, beta);
            }
        }

        // a^m = g^{ml} a_l (contravariant tangents).
        std::array<Eigen::RowVector<T, 3>, d> aup;
        for (std::size_t m = 0; m < d; ++m)
        {
            aup[m].setZero();
            for (std::size_t l = 0; l < d; ++l)
                aup[m] += g_inv_full(m, l) * a(l).row(q);
        }

        // Γ^m_{αβ} = a^m · a_{αβ}.
        for (std::size_t m = 0; m < d; ++m)
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = i; j < d; ++j)
                {
                    const Index packed = m * n_d2 + pack2<d>(i, j);
                    chr.Gamma_data(q, packed) = aup[m].dot(a_d1(i, j).row(q));
                }

        if (ord < 3) continue;

        // ∂_w g_{uv}: stored densely in dg[w], symmetric in (u, v).
        std::array<Eigen::Matrix<T, d, d>, d> dg;
        for (std::size_t w = 0; w < d; ++w) {
            for (std::size_t u = 0; u < d; ++u) {
                for (std::size_t v = u; v < d; ++v)
                {
                    const T val = a_d1(u, w).row(q).dot(a(v).row(q))
                                + a(u).row(q).dot(a_d1(v, w).row(q));
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
                    daup[m][gam] += dginv[gam](m, l) * a(l).row(q);
                    daup[m][gam] += g_inv_full(m, l) * a_d1(l, gam).row(q);
                }
            }
        }

        // ∂_γ Γ^m_{αβ} = (∂_γ a^m) · a_{αβ} + a^m · a_{αβγ}.
        for (std::size_t m = 0; m < d; ++m)
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = i; j < d; ++j)
                    for (std::size_t gam = 0; gam < d; ++gam)
                    {
                        const Index packed = m * (n_d2 * d) + pack2<d>(i, j) * d + gam;
                        chr.Gamma_d1_data(q, packed) =
                            daup[m][gam].dot(a_d1(i, j).row(q))
                          + aup[m].dot(a_d2(i, j, gam).row(q));
                    }
    }
}

// === Template Instantiations ========================================================

template class IntrinsicGeometry<double, 1>;
template class IntrinsicGeometry<double, 2>;
template class IntrinsicGeometry<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class IntrinsicGeometry<float, 1>;
template class IntrinsicGeometry<float, 2>;
template class IntrinsicGeometry<float, 3>;
#endif

} // namespace pyck
