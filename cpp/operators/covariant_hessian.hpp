#ifndef PYCK_COVARIANT_HESSIAN_HPP
#define PYCK_COVARIANT_HESSIAN_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "../elements/element_values.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Per-(quadrature-point, basis-function) differential-geometry operators.
 *
 * Each operator is a non-owning view bound to one quadrature point: it holds an
 * `ElementValues` reference and the point index `q`, and owns only the small per-qp
 * scratch it precomputes (Christoffels, permutation factors, connection derivatives).
 * The geometry stays owned by `ElementValues`; nothing large is copied. Construct on
 * the stack inside the element's q-loop — `Op op{ev, q}` — and reuse across the basis
 * loop, where each `operator()` is a pure contraction.
 */
namespace operators
{

/**
 * @brief Covariant Hessian @f$ H_{i\alpha\beta} = N^i_{,\alpha\beta}
 *        - \Gamma^\lambda_{\alpha\beta} N^i_{,\lambda} @f$. The second-kind Christoffel is
 *        raised from the cached first kind once in the ctor (hoisted out of the basis
 *        loop); each call is then just the contraction.
 */
template <std::floating_point T, std::size_t d>
struct CovariantHessian
{
    static constexpr std::size_t n_d2 = d * (d + 1) / 2;

    const ElementValues<T, d>&         ev;
    Index                              q;
    std::array<std::array<T, n_d2>, d> Gamma;   ///< Γ^λ_{(αβ)=c}, raised in the ctor.

    /// @brief Build at point @p q, raising the cached first-kind Christoffel into Γ^λ.
    CovariantHessian(const ElementValues<T, d>& ev, Index q) : ev(ev), q(q)
    {
        for (std::size_t lam = 0; lam < d; ++lam)
            for (std::size_t c = 0; c < n_d2; ++c) {
                T g = T(0);
                for (std::size_t e = 0; e < d; ++e)
                    g += ev.metric_inv(q, pack2<d>(lam, e))
                           * ev.christoffel_first(q, static_cast<Index>(e * n_d2 + c));
                Gamma[lam][c] = g;
            }
    }

    /// @brief Hessian for the Voigt-packed pair @p c.
    T packed(Index i, Index c) const
    {
        T h = ev.basis_derivs[2](i * static_cast<Index>(n_d2) + c, q);
        for (std::size_t lam = 0; lam < d; ++lam)
            h -= Gamma[lam][c]
                   * ev.basis_derivs[1](i * static_cast<Index>(d) + static_cast<Index>(lam), q);
        return h;
    }

    /// @brief Covariant Hessian @f$ H_{i\alpha\beta} @f$. Symmetric in (α, β).
    T operator()(Index i, Index alpha, Index beta) const
    { return packed(i, pack2<d>(alpha, beta)); }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_COVARIANT_HESSIAN_HPP
