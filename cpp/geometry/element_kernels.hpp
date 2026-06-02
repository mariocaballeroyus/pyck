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
 * connection is formed in place from base-vector dot products (raised by the
 * inverse metric where the second kind is needed) — no Christoffel array is
 * stored or consumed. Each `eval_*` returns a single per-basis kernel scalar,
 * computed on the fly from the cached position derivatives + metric and the raw
 * basis buffers `results_`; `Value` (= results[0]) and the covariant scalar
 * gradient (= results[1]) are already raw basis buffers.
 *
 * The lone exception is the Laplace–Beltrami *gradient* `P`, whose inverse-metric
 * and connection-trace gradients (`LaplaceGradAux`) are a genuinely expensive
 * 3rd-order quantity with no per-basis factor — the caller caches those once per
 * element and passes them in.
 */
namespace geometry::kernels
{

/**
 * @brief Covariant Hessian of basis @p i for the Voigt-packed pair @p p:
 *        @f$ H_{ip} = N^i_{,p} - \Gamma^\mu_{p}\, N^i_{,\mu} @f$, with the
 *        second-kind connection @f$ \Gamma^\mu_{p} = g^{\mu\varepsilon}
 *        (\mathbf{A}_\varepsilon\cdot\mathbf{A}_{,p}) @f$ formed in place.
 */
template <std::floating_point T, std::size_t d>
inline T
eval_hessian_packed(const std::vector<Matrix<T>>& results,
                    const std::vector<ColMatrix<T, 3>>& position_data,
                    const Matrix<T>& g_inv_data,
                    Index i, Index p, Index q)
{
    constexpr std::size_t n_d2 = d * (d + 1) / 2;
    const Index Q = g_inv_data.rows();

    std::array<T, d> first{};   // first[ε] = A_ε · A_{,p}
    for (std::size_t e = 0; e < d; ++e)
        first[e] = position_data[1].row(static_cast<Index>(e) * Q + q)
                     .dot(position_data[2].row(p * Q + q));

    T h = results[2](i * static_cast<Index>(n_d2) + p, q);
    for (std::size_t m = 0; m < d; ++m) {
        T gam = T(0);   // Γ^μ_p
        for (std::size_t e = 0; e < d; ++e)
            gam += g_inv_data(q, pack2<d>(m, e)) * first[e];
        h -= gam * results[1](i * static_cast<Index>(d) + static_cast<Index>(m), q);
    }
    return h;
}

/// @brief Covariant Hessian @f$ H_{i\alpha\beta} @f$. Symmetric in (α, β).
template <std::floating_point T, std::size_t d>
inline T
eval_hessian(const std::vector<Matrix<T>>& results,
             const std::vector<ColMatrix<T, 3>>& position_data,
             const Matrix<T>& g_inv_data,
             Index i, Index alpha, Index beta, Index q)
{
    return eval_hessian_packed<T, d>(results, position_data, g_inv_data,
                                     i, pack2<d>(alpha, beta), q);
}

/**
 * @brief Laplace–Beltrami @f$ L_i = g^{\alpha\beta} H_{i\alpha\beta} @f$
 *        (off-diagonal Voigt components doubled).
 */
template <std::floating_point T, std::size_t d>
inline T
eval_laplace_beltrami(const std::vector<Matrix<T>>& results,
                      const std::vector<ColMatrix<T, 3>>& position_data,
                      const Matrix<T>& g_inv_data,
                      Index i, Index q)
{
    constexpr std::size_t n_d2 = d * (d + 1) / 2;
    T l = T(0);
    for (std::size_t p = 0; p < n_d2; ++p) {
        const T w = (p < d) ? T(1) : T(2);   // Voigt: diagonals first
        l += w * g_inv_data(q, static_cast<Index>(p))
               * eval_hessian_packed<T, d>(results, position_data, g_inv_data,
                                           i, static_cast<Index>(p), q);
    }
    return l;
}

/**
 * @brief Covariant gradient of a contravariant in-plane vector field:
 *        @f$ D_{i\lambda\alpha\beta} = A_{\alpha\lambda} N^i_{,\beta}
 *            + (\mathbf{A}_\alpha\cdot\mathbf{A}_{\lambda,\beta}) N^i @f$,
 *        so that @f$ u_{\alpha|\beta} = D_{i\lambda\alpha\beta}\, u^{\lambda}_i @f$.
 *        Self-contained from base vectors (covariant metric + first-kind
 *        connection, both plain dot products).
 */
template <std::floating_point T, std::size_t d>
inline T
eval_vector_gradient(const std::vector<Matrix<T>>& results,
                     const std::vector<ColMatrix<T, 3>>& position_data,
                     Index i, Index lambda, Index alpha, Index beta, Index q)
{
    constexpr Index dd = static_cast<Index>(d);
    const Index Q = results[0].cols();
    const T g_al = position_data[1].row(alpha * Q + q)
                     .dot(position_data[1].row(lambda * Q + q));
    const T conn = position_data[1].row(alpha * Q + q)
                     .dot(position_data[2].row(pack2<d>(lambda, beta) * Q + q));
    return g_al * results[1](i * dd + beta, q) + conn * results[0](i, q);
}

/**
 * @brief Surface gradient of the Laplace–Beltrami,
 *        @f$ P_{i\alpha} = \partial_\alpha(g^{\mu\nu} H_{i\mu\nu}) @f$ — a genuine
 *        3rd-order kernel (needs `results` order 3). Inverse-metric / connection
 *        gradients arrive via `aux` (`compute_laplace_grad_aux`).
 */
template <std::floating_point T, std::size_t d>
inline T
eval_lb_gradient(const std::vector<Matrix<T>>& results,
                 const LaplaceGradAux<T, d>& aux,
                 const Matrix<T>& g_inv_data,
                 Index i, Index alpha, Index q)
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

} // namespace geometry::kernels

} // namespace pyck

#endif // PYCK_ELEMENT_KERNELS_HPP
