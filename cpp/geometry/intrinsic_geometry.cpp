#include "intrinsic_geometry.hpp"

#include <Eigen/Core>
#include <cmath>

namespace pyck
{

template <std::floating_point T, std::size_t d>
IntrinsicGeometry<T, d>::IntrinsicGeometry(const BasisValues<T, d>& basis,
                                           const ColMatrix<T, 3>& act_pts)
{
    const Index order_b = basis.order();
    const Index Q_      = basis.Q();
    const Index N_      = basis.N();
    constexpr Index n_metric = d * (d + 1) / 2;

    // --- Allocate position storage per order (data_[k] of shape (Q · n_k) × 3) ------

    position_data.resize(order_b + 1);
    for (Index k = 0; k <= order_b; ++k) {
        const Index n_k = num_multi_indices<d>(k);
        position_data[k].resize(Q_ * n_k, 3);
    }
    g_data.resize(Q_, n_metric);
    g_inv_data.resize(Q_, n_metric);
    jac.resize(Q_);

    for (Index q = 0; q < Q_; ++q)
    {
        // --- Position + all derivatives at q ----------------------------------------

        for (Index k = 0; k <= order_b; ++k)
        {
            const Index n_k = num_multi_indices<d>(k);
            auto slab = basis.data()[k].col(q);     // length N · n_k
            for (Index packed = 0; packed < n_k; ++packed) {
                Eigen::Matrix<T, 1, 3> deriv_q = Eigen::Matrix<T, 1, 3>::Zero();
                for (Index b = 0; b < N_; ++b) {
                    deriv_q.noalias() += slab(b * n_k + packed) * act_pts.row(b);
                }
                position_data[k].row(packed * Q_ + q) = deriv_q;
            }
        }

        // --- Metric + jacobian (requires order ≥ 1; uses position_data[1]) ----------

        if (order_b < 1) continue;

        Eigen::Matrix<T, d, d> g_mat;
        for (Index i = 0; i < d; ++i)
            for (Index j = i; j < d; ++j) {
                const T g_ij = a(i).row(q).dot(a(j).row(q));
                g_mat(i, j) = g_ij;
                if (j != i) g_mat(j, i) = g_ij;
            }
        const Eigen::Matrix<T, d, d> inv_g = g_mat.inverse();
        const T det_g = g_mat.determinant();
        for (Index i = 0; i < d; ++i)
            for (Index j = i; j < d; ++j) {
                const Index packed = pack2<d>(i, j);
                g_data    (q, packed) = g_mat(i, j);
                g_inv_data(q, packed) = inv_g(i, j);
            }
        jac(q) = std::sqrt(det_g);
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
