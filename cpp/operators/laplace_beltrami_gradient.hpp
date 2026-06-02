#ifndef PYCK_LAPLACE_BELTRAMI_GRADIENT_HPP
#define PYCK_LAPLACE_BELTRAMI_GRADIENT_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

namespace operators
{

/**
 * @brief Per-quadrature-point *connector* for the Laplace–Beltrami gradient: the
 *        inverse-metric gradient and connection-trace gradients that ∂_α(Δ_g f)
 *        needs. Plain data — owned by the caller (cached once per element in
 *        `ElementValues::lb_grad_conn_`) and referenced by the @ref LaplaceBeltramiGradient
 *        operator. This is the one genuinely 3rd-order, basis-independent quantity
 *        worth caching; the cheaper connections are formed inline by their operators.
 *
 * Storage matches the upper-triangular convention of the geometry primitives:
 * symmetric tensor indices fill only their i ≤ j slots. Each `Vector<T>` is
 * per-quadrature.
 */
template <std::floating_point T, std::size_t d>
struct LaplaceBeltramiGradConn
{
    /// (g^{ij})_{,α}, symmetric in (i, j); only i ≤ j filled. G_inv_d[i][j][α].
    std::array<std::array<std::array<Vector<T>, d>, d>, d> G_inv_d;

    /// c^δ = g^{ij} Γ^δ_{ij}.                                   c[δ].
    std::array<Vector<T>, d> c;

    /// (c^δ)_{,α}.                                              c_d[δ][α].
    std::array<std::array<Vector<T>, d>, d> c_d;
};

/**
 * @brief Build a LaplaceBeltramiGradConn<T, d> at all quadrature points from raw base-vector
 *        data: the position derivatives (must be populated to order 3 — the basis
 *        evaluated at order ≥ 3) and the contravariant metric. The connection is
 *        formed internally from base-vector dot products `A_ε·A_{,ij}` raised by
 *        `g_inv` — no Christoffel array is needed.
 */
template <std::floating_point T, std::size_t d>
LaplaceBeltramiGradConn<T, d>
compute_laplace_beltrami_grad_conn(const std::vector<ColMatrix<T, 3>>& position_data,
                         const Matrix<T>& g_inv_data);

/**
 * @brief Surface gradient of the Laplace–Beltrami,
 *        @f$ P_{i\alpha} = \partial_\alpha(g^{\mu\nu} H_{i\mu\nu}) @f$ — a genuine
 *        3rd-order operator (needs `results` order 3). A non-owning view that
 *        references the basis buffers, the contravariant metric, and the cached
 *        @ref LaplaceBeltramiGradConn connector (all workspace-owned).
 */
template <std::floating_point T, std::size_t d>
struct LaplaceBeltramiGradient
{
    const std::vector<Matrix<T>>& results;
    const LaplaceBeltramiGradConn<T, d>&   aux;
    const Matrix<T>&              g_inv_data;

    /// @brief Laplace–Beltrami gradient @f$ P_{i\alpha}(q) @f$ of basis function i.
    T operator()(Index i, Index alpha, Index q) const
    {
        constexpr std::size_t n_d2 = d * (d + 1) / 2;
        constexpr std::size_t n_d3 = static_cast<std::size_t>(num_multi_indices<d>(3));
        const std::size_t a = static_cast<std::size_t>(alpha);

        T p = T(0);
        // Σ_{μ≤ν} w[(g^{μν})_,α N_{,μν} + g^{μν} N_{,μνα}]
        for (std::size_t mu = 0; mu < d; ++mu)
            for (std::size_t nu = mu; nu < d; ++nu) {
                const T w = (mu == nu) ? T(1) : T(2);
                const Index p2 = pack2<d>(mu, nu);
                const Index p3 = pack3<d>(mu, nu, a);
                p += w * aux.G_inv_d[mu][nu][a](q)
                       * results[2](i * static_cast<Index>(n_d2) + p2, q);
                p += w * g_inv_data(q, p2)
                       * results[3](i * static_cast<Index>(n_d3) + p3, q);
            }
        // − Σ_δ [(c^δ)_,α N_{,δ} + c^δ N_{,δα}]
        for (std::size_t delta = 0; delta < d; ++delta) {
            p -= aux.c_d[delta][a](q)
                   * results[1](i * static_cast<Index>(d) + static_cast<Index>(delta), q);
            p -= aux.c[delta](q)
                   * results[2](i * static_cast<Index>(n_d2) + pack2<d>(delta, a), q);
        }
        return p;
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_GRADIENT_HPP
