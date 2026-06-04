#ifndef PYCK_LAPLACE_BELTRAMI_GRADIENT_HPP
#define PYCK_LAPLACE_BELTRAMI_GRADIENT_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../elements/element_values.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

namespace operators
{

/**
 * @brief Per-quadrature-point connector for the Laplace–Beltrami gradient: the
 *        inverse-metric gradient and connection-trace gradients that ∂_α(Δ_g f) needs.
 *        A small fixed-size POD — the genuinely 3rd-order, basis-independent quantity —
 *        built once per qp on the stack and held by the @ref LaplaceBeltramiGradient
 *        operator.
 *
 * Storage matches the upper-triangular convention of the geometry primitives:
 * symmetric tensor indices fill only their i ≤ j slots.
 */
template <std::floating_point T, std::size_t d>
struct LaplaceBeltramiGradConn
{
    /// (A^{ij})_{,α}, symmetric in (i, j); only i ≤ j filled. G_inv_d[i][j][α].
    std::array<std::array<std::array<T, d>, d>, d> G_inv_d{};

    /// c^δ = A^{ij} Γ^δ_{ij}.                                   c[δ].
    std::array<T, d> c{};

    /// (c^δ)_{,α}.                                              c_d[δ][α].
    std::array<std::array<T, d>, d> c_d{};
};

/**
 * @brief Build a LaplaceBeltramiGradConn<T, d> at quadrature point @p q. The order-2
 *        connection terms (`A_ε·A_{,ij}` and the second-kind Christoffel) are seeded from
 *        the cached first-kind Christoffel buffer; the genuinely 3rd-order terms
 *        (`A_{,ij}·A_{,kl}`, `A_ε·A_{,ijk}` and their combinations) are formed here from
 *        the position derivatives (populated to order 3) and the contravariant metric.
 *        Defined out-of-line (explicit instantiation) so the heavy build is compiled once
 *        rather than inlined into every consumer.
 */
template <std::floating_point T, std::size_t d>
LaplaceBeltramiGradConn<T, d>
compute_laplace_beltrami_grad_conn(const Matrix<T>& christoffel_first,
                                   const std::vector<ColMatrix<T, 3>>& position_derivs,
                                   const Matrix<T>& metric_inv, Index q);

/**
 * @brief Surface gradient of the Laplace–Beltrami,
 *        @f$ P_{i\alpha} = \partial_\alpha(A^{\mu\nu} H_{i\mu\nu}) @f$ — a genuine
 *        3rd-order operator (needs `basis_derivs` order 3). Non-owning view: builds its
 *        @ref LaplaceBeltramiGradConn connector once for the fixed qp in the constructor
 *        (heavy build out-of-line), then each call is a contraction.
 */
template <std::floating_point T, std::size_t d>
struct LaplaceBeltramiGradient
{
    const ElementValues<T, d>&    ev;
    Index                         q;
    LaplaceBeltramiGradConn<T, d> aux;

    /// @brief Build at point @p q.
    LaplaceBeltramiGradient(const ElementValues<T, d>& ev, Index q)
        : ev(ev), q(q),
          aux(compute_laplace_beltrami_grad_conn<T, d>(ev.christoffel_first,
                                                       ev.position_derivs,
                                                       ev.metric_inv_, q))
    {}

    /// @brief Laplace–Beltrami gradient @f$ P_{i\alpha} @f$ of basis function i.
    T operator()(Index i, Index alpha) const
    {
        constexpr std::size_t n_d2 = d * (d + 1) / 2;
        constexpr std::size_t n_d3 = static_cast<std::size_t>(num_multi_indices<d>(3));
        const std::size_t a = static_cast<std::size_t>(alpha);

        T out = T(0);
        // Σ_{μ≤ν} w[(A^{μν})_,α N_{,μν} + A^{μν} N_{,μνα}]
        for (std::size_t mu = 0; mu < d; ++mu)
            for (std::size_t nu = mu; nu < d; ++nu) {
                const T w = (mu == nu) ? T(1) : T(2);
                const Index p2 = pack2<d>(mu, nu);
                const Index p3 = pack3<d>(mu, nu, a);
                out += w * aux.G_inv_d[mu][nu][a]
                         * ev.basis_derivs[2](i * static_cast<Index>(n_d2) + p2, q);
                out += w * ev.metric_inv(q, p2)
                         * ev.basis_derivs[3](i * static_cast<Index>(n_d3) + p3, q);
            }
        // − Σ_δ [(c^δ)_,α N_{,δ} + c^δ N_{,δα}]
        for (std::size_t delta = 0; delta < d; ++delta) {
            out -= aux.c_d[delta][a]
                     * ev.basis_derivs[1](i * static_cast<Index>(d) + static_cast<Index>(delta), q);
            out -= aux.c[delta]
                     * ev.basis_derivs[2](i * static_cast<Index>(n_d2) + pack2<d>(delta, a), q);
        }
        return out;
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_GRADIENT_HPP
