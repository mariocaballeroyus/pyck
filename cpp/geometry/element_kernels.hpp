#ifndef PYCK_ELEMENT_KERNELS_HPP
#define PYCK_ELEMENT_KERNELS_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "laplace_beltrami.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Per-(quadrature-point, basis-function) differential-geometry kernels.
 *
 * Each strain matrix assembles as `Σ (per-qp geometric coefficient) × (kernel)`.
 * Kernels are reference-geometry only (material-agnostic) and basis-direct: the
 * connection is formed from base-vector dot products raised by the inverse
 * metric — no Christoffel array is consumed. `Value` (= results[0]) and the
 * covariant scalar gradient (= results[1]) are already the raw basis buffers, so
 * only the Hessian, vector gradient, Laplace–Beltrami, and its gradient need
 * materialising here.
 *
 * Output packing mirrors `results_[k]`: `data.col(q)(i·n_comp + packed)`, with
 * symmetric (α,β) pairs in Voigt order via `pack2<d>` (diagonals first).
 */
namespace geometry::kernels
{

/**
 * @brief Covariant Hessian of each scalar basis function:
 *        @f$ H_{i\alpha\beta} = N^i_{,\alpha\beta}
 *            - \Gamma^\mu_{\alpha\beta}\, N^i_{,\mu} @f$,
 *        with the connection built directly from base vectors,
 *        @f$ \Gamma^\mu_{\alpha\beta} = g^{\mu\varepsilon}
 *            (\mathbf{A}_\varepsilon\cdot\mathbf{A}_{,\alpha\beta}) @f$.
 *
 * @param results        Basis buffers; needs orders 1 and 2.
 * @param position_data  Surface position derivatives (orders 1 and 2).
 * @param g_inv_data     Contravariant metric, Voigt-packed (Q × n_d2).
 * @param H_data         Out: (N·n_d2, Q), packed `i·n_d2 + pack2<d>(α,β)`.
 */
template <std::floating_point T, std::size_t d>
inline void
compute_kernel_hessian(const std::vector<Matrix<T>>& results,
                       const std::vector<ColMatrix<T, 3>>& position_data,
                       const Matrix<T>& g_inv_data,
                       Matrix<T>& H_data)
{
    constexpr std::size_t n_d2 = d * (d + 1) / 2;
    const Index Q = results[0].cols();
    const Index N = results[0].rows();
    H_data.resize(N * static_cast<Index>(n_d2), Q);

    for (Index q = 0; q < Q; ++q)
    {
        // Γ^μ_{αβ} = g^{με}(A_ε·A_{,αβ}), shared across all basis functions.
        std::array<std::array<T, n_d2>, d> Gam{};
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t b = a; b < d; ++b) {
                const Index pab = pack2<d>(a, b);
                std::array<T, d> first{};   // first[ε] = A_ε · A_{,αβ}
                for (std::size_t e = 0; e < d; ++e)
                    first[e] = position_data[1].row(static_cast<Index>(e) * Q + q)
                                 .dot(position_data[2].row(pab * Q + q));
                for (std::size_t m = 0; m < d; ++m) {
                    T s = T(0);
                    for (std::size_t e = 0; e < d; ++e)
                        s += g_inv_data(q, pack2<d>(m, e)) * first[e];
                    Gam[m][static_cast<std::size_t>(pab)] = s;
                }
            }

        const auto N1 = results[1].col(q);   // N^i_{,μ}  at (i·d + μ)
        const auto N2 = results[2].col(q);   // N^i_{,αβ} at (i·n_d2 + pack2)
        for (Index i = 0; i < N; ++i)
            for (std::size_t p = 0; p < n_d2; ++p) {
                T h = N2(i * static_cast<Index>(n_d2) + static_cast<Index>(p));
                for (std::size_t m = 0; m < d; ++m)
                    h -= Gam[m][p] * N1(i * static_cast<Index>(d) + static_cast<Index>(m));
                H_data(i * static_cast<Index>(n_d2) + static_cast<Index>(p), q) = h;
            }
    }
}

/**
 * @brief Laplace–Beltrami of each scalar basis function:
 *        @f$ L_i = g^{\alpha\beta} H_{i\alpha\beta} @f$ — the inverse-metric
 *        contraction of the covariant Hessian (off-diagonal Voigt components
 *        doubled). Run after `compute_kernel_hessian`.
 *
 * @param H_data      Covariant Hessian kernel, (N·n_d2, Q).
 * @param g_inv_data  Contravariant metric, Voigt-packed (Q × n_d2).
 * @param L_data      Out: (N, Q).
 */
template <std::floating_point T, std::size_t d>
inline void
compute_kernel_laplace_beltrami(const Matrix<T>& H_data,
                                const Matrix<T>& g_inv_data,
                                Matrix<T>& L_data)
{
    constexpr std::size_t n_d2 = d * (d + 1) / 2;
    const Index Q = H_data.cols();
    const Index N = H_data.rows() / static_cast<Index>(n_d2);
    L_data.resize(N, Q);

    for (Index q = 0; q < Q; ++q)
        for (Index i = 0; i < N; ++i) {
            T l = T(0);
            for (std::size_t p = 0; p < n_d2; ++p) {
                const T w = (p < d) ? T(1) : T(2);   // Voigt: diagonals first
                l += w * g_inv_data(q, static_cast<Index>(p))
                       * H_data(i * static_cast<Index>(n_d2) + static_cast<Index>(p), q);
            }
            L_data(i, q) = l;
        }
}

/**
 * @brief Covariant gradient of a contravariant in-plane vector field, per basis:
 *        @f$ D_{i\lambda\alpha\beta} = A_{\alpha\lambda} N^i_{,\beta}
 *            + (\mathbf{A}_\alpha\cdot\mathbf{A}_{\lambda,\beta}) N^i @f$,
 *        so that @f$ u_{\alpha|\beta} = D_{i\lambda\alpha\beta}\, u^{\lambda}_i @f$.
 *        Self-contained from base vectors (covariant metric + first-kind
 *        connection); no inverse metric or Christoffel array.
 *
 * @param results        Basis buffers; needs orders 0 and 1.
 * @param position_data  Surface position derivatives (orders 1 and 2).
 * @param D_data         Out: dense (N·d³, Q), packed `i·d³ + (λ·d + α)·d + β`.
 */
template <std::floating_point T, std::size_t d>
inline void
compute_kernel_vector_gradient(const std::vector<Matrix<T>>& results,
                               const std::vector<ColMatrix<T, 3>>& position_data,
                               Matrix<T>& D_data)
{
    constexpr std::size_t nc = d * d * d;
    const Index Q = results[0].cols();
    const Index N = results[0].rows();
    D_data.resize(N * static_cast<Index>(nc), Q);

    for (Index q = 0; q < Q; ++q)
    {
        // g_{αλ} = A_α·A_λ ; conn[α][λ][β] = A_α·A_{λ,β}. Shared across basis.
        std::array<std::array<T, d>, d> g{};
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t l = a; l < d; ++l) {
                const T v = position_data[1].row(static_cast<Index>(a) * Q + q)
                              .dot(position_data[1].row(static_cast<Index>(l) * Q + q));
                g[a][l] = v;
                g[l][a] = v;
            }
        std::array<std::array<std::array<T, d>, d>, d> conn{};
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t l = 0; l < d; ++l)
                for (std::size_t b = 0; b < d; ++b)
                    conn[a][l][b] = position_data[1].row(static_cast<Index>(a) * Q + q)
                                      .dot(position_data[2].row(pack2<d>(l, b) * Q + q));

        const auto N0 = results[0].col(q);
        const auto N1 = results[1].col(q);
        for (Index i = 0; i < N; ++i) {
            const T Ni = N0(i);
            for (std::size_t l = 0; l < d; ++l)
                for (std::size_t a = 0; a < d; ++a)
                    for (std::size_t b = 0; b < d; ++b) {
                        const std::size_t c = (l * d + a) * d + b;
                        D_data(i * static_cast<Index>(nc) + static_cast<Index>(c), q)
                            = g[a][l] * N1(i * static_cast<Index>(d) + static_cast<Index>(b))
                            + conn[a][l][b] * Ni;
                    }
        }
    }
}

/**
 * @brief Surface gradient of the Laplace–Beltrami of each scalar basis function,
 *        @f$ P_{i\alpha} = \partial_\alpha(g^{\mu\nu} H_{i\mu\nu}) @f$. A genuine
 *        3rd-order kernel (needs `results` order 3) — it cannot be obtained by
 *        differencing the discrete `H` field. The inverse-metric and connection
 *        gradients arrive via `aux` (see `compute_laplace_grad_aux`). The
 *        material `ratio` is NOT included — the element forms `−(K_b/K_s) P`.
 *
 * @param results      Basis buffers; needs orders 1, 2, 3.
 * @param aux          Inverse-metric / connection-trace gradients.
 * @param g_inv_data   Contravariant metric, Voigt-packed (Q × n_d2).
 * @param P_data       Out: (N·d, Q), packed `i·d + α`.
 */
template <std::floating_point T, std::size_t d>
inline void
compute_kernel_lb_gradient(const std::vector<Matrix<T>>& results,
                           const LaplaceGradAux<T, d>& aux,
                           const Matrix<T>& g_inv_data,
                           Matrix<T>& P_data)
{
    constexpr std::size_t n_d2 = d * (d + 1) / 2;
    constexpr std::size_t n_d3 = static_cast<std::size_t>(num_multi_indices<d>(3));
    const Index Q = results[0].cols();
    const Index N = results[0].rows();
    P_data.resize(N * static_cast<Index>(d), Q);

    for (Index q = 0; q < Q; ++q)
    {
        const auto N1 = results[1].col(q);   // N^i_{,δ}    (i·d + δ)
        const auto N2 = results[2].col(q);   // N^i_{,μν}   (i·n_d2 + pack2)
        const auto N3 = results[3].col(q);   // N^i_{,μνα}  (i·n_d3 + pack3)
        for (Index i = 0; i < N; ++i)
            for (std::size_t alpha = 0; alpha < d; ++alpha) {
                T p = T(0);
                // Σ_{μ≤ν} w[(g^{μν})_,α N_{,μν} + g^{μν} N_{,μνα}]
                for (std::size_t mu = 0; mu < d; ++mu)
                    for (std::size_t nu = mu; nu < d; ++nu) {
                        const T w = (mu == nu) ? T(1) : T(2);
                        const Index p2 = pack2<d>(mu, nu);
                        const Index p3 = pack3<d>(mu, nu, alpha);
                        p += w * aux.G_inv_d[mu][nu][alpha](q)
                               * N2(i * static_cast<Index>(n_d2) + p2);
                        p += w * g_inv_data(q, p2)
                               * N3(i * static_cast<Index>(n_d3) + p3);
                    }
                // − Σ_δ [(c^δ)_,α N_{,δ} + c^δ N_{,δα}]
                for (std::size_t delta = 0; delta < d; ++delta) {
                    p -= aux.c_d[delta][alpha](q) * N1(i * static_cast<Index>(d) + static_cast<Index>(delta));
                    p -= aux.c[delta](q)
                           * N2(i * static_cast<Index>(n_d2) + pack2<d>(delta, alpha));
                }
                P_data(i * static_cast<Index>(d) + static_cast<Index>(alpha), q) = p;
            }
    }
}

} // namespace geometry::kernels

} // namespace pyck

#endif // PYCK_ELEMENT_KERNELS_HPP
