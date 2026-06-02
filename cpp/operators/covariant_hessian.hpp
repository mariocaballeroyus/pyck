#ifndef PYCK_COVARIANT_HESSIAN_HPP
#define PYCK_COVARIANT_HESSIAN_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Per-(quadrature-point, basis-function) differential-geometry operators.
 *
 * Each operator is a lightweight view bound to one element's cached primitive
 * buffers (`{ev.results_, ev.position_data, ev.g_inv_data, …}`), exposing a single
 * `operator()` so call sites read `hess(i, α, β, q)`. Reference-geometry only and
 * basis-direct: the connection is formed in place from base-vector dot products
 * (raised by the inverse metric where the second kind is needed). The cheap
 * operators hold no connector; the genuinely 3rd-order @ref LaplaceBeltramiGradient
 * binds a cached @ref LaplaceBeltramiGradConn. Views depend on the buffer types, not on
 * `ElementValues`.
 */
namespace operators
{

/**
 * @brief Covariant Hessian @f$ H_{i\alpha\beta} = N^i_{,\alpha\beta}
 *        - \Gamma^\mu_{\alpha\beta} N^i_{,\mu} @f$, with the second-kind connection
 *        @f$ \Gamma^\mu_{\alpha\beta} = g^{\mu\varepsilon}
 *        (\mathbf{A}_\varepsilon\cdot\mathbf{A}_{,\alpha\beta}) @f$ formed in place.
 */
template <std::floating_point T, std::size_t d>
struct CovariantHessian
{
    const std::vector<Matrix<T>>&       results;
    const std::vector<ColMatrix<T, 3>>& position_data;
    const Matrix<T>&                    g_inv_data;

    /// @brief Hessian for the Voigt-packed pair @p p (used by @ref LaplaceBeltrami).
    T packed(Index i, Index p, Index q) const
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

    /// @brief Covariant Hessian @f$ H_{i\alpha\beta}(q) @f$. Symmetric in (α, β).
    T operator()(Index i, Index alpha, Index beta, Index q) const
    { return packed(i, pack2<d>(alpha, beta), q); }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_COVARIANT_HESSIAN_HPP
