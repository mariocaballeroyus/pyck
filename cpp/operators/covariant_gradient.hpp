#ifndef PYCK_COVARIANT_GRADIENT_HPP
#define PYCK_COVARIANT_GRADIENT_HPP

#include <concepts>
#include <cstddef>

#include "../elements/element_values.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

namespace operators
{

/**
 * @brief Covariant gradient of a contravariant in-plane vector field:
 *        @f$ D_{i\lambda\alpha\beta} = A_{\alpha\lambda} N^i_{,\beta}
 *            + \Gamma_{\alpha,(\lambda\beta)} N^i @f$,
 *        so that @f$ u_{\alpha|\beta} = D_{i\lambda\alpha\beta}\, u^{\lambda}_i @f$.
 *
 * Reads the covariant metric @f$ A_{\alpha\lambda} @f$ and the first-kind Christoffel
 * @f$ \Gamma_{\alpha,(\lambda\beta)} = \mathbf{A}_\alpha\cdot\mathbf{A}_{\lambda,\beta} @f$
 * straight from the cached buffers — no inverse metric, no symbol re-formed. Non-owning
 * view; each call is a contraction.
 */
template <std::floating_point T, std::size_t d>
struct CovariantGradient
{
    static constexpr std::size_t n_d2 = d * (d + 1) / 2;

    const ElementValues<T, d>& ev;
    Index                      q;

    CovariantGradient(const ElementValues<T, d>& ev, Index q) : ev(ev), q(q) {}

    /// @brief @f$ D_{i\lambda\alpha\beta} @f$.
    T operator()(Index i, Index lambda, Index alpha, Index beta) const
    {
        constexpr Index dd = static_cast<Index>(d);
        return ev.metric(q, pack2<d>(alpha, lambda)) * ev.basis_derivs[1](i * dd + beta, q)
             + ev.christoffel_first(q, alpha * static_cast<Index>(n_d2) + pack2<d>(lambda, beta))
                 * ev.basis_derivs[0](i, q);
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_COVARIANT_GRADIENT_HPP
