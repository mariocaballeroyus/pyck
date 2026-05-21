#include "intrinsic_geometry.hpp"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace pyck
{

template <std::floating_point T, std::size_t d>
void IntrinsicGeometry<T, d>::reinit(const std::vector<Matrix<T>>& basis,
                                     const ColMatrix<T, 3>& act_pts,
                                     IntrinsicGeometryFlags flags)
{
    const Index order_b = static_cast<Index>(basis.size()) - 1;
    const Index Q_      = basis[0].cols();
    const Index N_      = basis[0].rows();
    constexpr Index n_metric = d * (d + 1) / 2;
    constexpr Index n_d2     = n_metric;

    // Best-effort: clamp flags to what the basis can provide. Callers using
    // default flags get the maximum available; explicit opt-outs still apply.
    if (order_b < 1) flags.metric          = false;
    if (order_b < 2) flags.christoffels    = false;
    if (order_b < 3) flags.christoffels_d1 = false;

    // --- Allocations (flag-gated; no-op when already sized) ---------------------------

    position_data.resize(order_b + 1);
    for (Index k = 0; k <= order_b; ++k) {
        const Index n_k = num_multi_indices<d>(k);
        position_data[k].resize(Q_ * n_k, 3);
    }
    if (flags.metric) {
        g_data.resize(Q_, n_metric);
        g_inv_data.resize(Q_, n_metric);
        jac.resize(Q_);
    }
    if (flags.christoffels)    Gamma_data.resize(Q_, d * n_d2);
    if (flags.christoffels_d1) Gamma_d1_data.resize(Q_, d * n_d2 * d);

    // --- Pass 1: position derivatives (always) ----------------------------------------

    for (Index q = 0; q < Q_; ++q) {
        for (Index k = 0; k <= order_b; ++k) {
            const Index n_k = num_multi_indices<d>(k);
            auto slab = basis[k].col(q);            // length N · n_k
            for (Index packed = 0; packed < n_k; ++packed) {
                Eigen::Matrix<T, 1, 3> deriv_q = Eigen::Matrix<T, 1, 3>::Zero();
                for (Index b = 0; b < N_; ++b) {
                    deriv_q.noalias() += slab(b * n_k + packed) * act_pts.row(b);
                }
                position_data[k].row(packed * Q_ + q) = deriv_q;
            } // derivative indices
        } // derivative orders
    } // quadrature pts

    // --- Pass 2: metric + jacobian ----------------------------------------------------

    if (!flags.metric) return;

    for (Index q = 0; q < Q_; ++q)
    {
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

    // --- Pass 3: Christoffels (+ Γ_d1 in the same loop iff requested) -----------------

    if (!flags.christoffels) return;

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
                    Gamma_data(q, packed) = aup[m].dot(a_d1(i, j).row(q));
                }

        if (!flags.christoffels_d1) continue;

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
                        Gamma_d1_data(q, packed) =
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
