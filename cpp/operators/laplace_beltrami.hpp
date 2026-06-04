#ifndef PYCK_LAPLACE_BELTRAMI_HPP
#define PYCK_LAPLACE_BELTRAMI_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "covariant_hessian.hpp"
#include "../elements/element_values.hpp"
#include "../types.hpp"

namespace pyck
{

namespace operators
{

/**
 * @brief Laplace–Beltrami operator @f$ L_i = A^{\alpha\beta} H_{i\alpha\beta} @f$
 *        — the inverse-metric contraction of the covariant Hessian (off-diagonal
 *        Voigt components doubled). Non-owning view: holds a @ref CovariantHessian (its
 *        connection raised in the ctor) plus the Voigt-weighted @f$ A^{\alpha\beta} @f$
 *        weights, both formed once for the fixed qp.
 */
template <std::floating_point T, std::size_t d>
struct LaplaceBeltrami
{
    static constexpr std::size_t n_d2 = d * (d + 1) / 2;

    CovariantHessian<T, d> hess;
    std::array<T, n_d2>    w;     ///< Voigt-weighted A^{αβ}: w[c] = (c<d?1:2)·A^{c}.

    /// @brief Build at point @p q.
    LaplaceBeltrami(const ElementValues<T, d>& ev, Index q) : hess(ev, q)
    {
        for (std::size_t c = 0; c < n_d2; ++c)
            w[c] = ((c < d) ? T(1) : T(2)) * ev.metric_inv(q, static_cast<Index>(c));
    }

    /// @brief Laplace–Beltrami @f$ L_i @f$ of basis function i.
    T operator()(Index i) const
    {
        T l = T(0);
        for (std::size_t c = 0; c < n_d2; ++c)
            l += w[c] * hess.packed(i, static_cast<Index>(c));
        return l;
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_HPP
