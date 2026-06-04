#ifndef PYCK_CURL_GRADIENT_HPP
#define PYCK_CURL_GRADIENT_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "covariant_hessian.hpp"
#include "../elements/element_values.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

namespace operators
{

/**
 * @brief Covariant gradient of the surface @ref Curl of a scalar basis function:
 *        @f$ (\mathrm{curl}\,N^i)_{\alpha|\beta}
 *            = \varepsilon_\alpha{}^\delta H_{i\delta\beta} @f$, where @f$ H @f$ is the
 *        @ref CovariantHessian and the permutation tensor is covariantly constant.
 *
 * Non-owning view: holds a @ref CovariantHessian (connection raised in its ctor) plus the
 * mixed permutation, both formed once for the fixed qp. Returns the (non-symmetric)
 * (α, β) component; consumers symmetrise as the strain measure requires. Surface
 * operator (d = 2).
 */
template <std::floating_point T, std::size_t d>
struct CurlGradient
{
    CovariantHessian<T, d>          hess;
    std::array<std::array<T, d>, d> eps;   ///< ε_α^β.

    /// @brief Build at point @p q.
    CurlGradient(const ElementValues<T, d>& ev, Index q) : hess(ev, q)
    {
        const T invJ = T(1) / ev.jac(q);
        for (std::size_t a = 0; a < d; ++a) {
            eps[a][0] = -ev.metric(q, pack2<d>(static_cast<Index>(a), 1)) * invJ;   // ε_α^0
            eps[a][1] =  ev.metric(q, pack2<d>(static_cast<Index>(a), 0)) * invJ;   // ε_α^1
        }
    }

    /// @brief @f$ \varepsilon_\alpha{}^\delta H_{i\delta\beta} @f$.
    T operator()(Index i, Index alpha, Index beta) const
    {
        const std::size_t a = static_cast<std::size_t>(alpha);
        return eps[a][0] * hess(i, 0, beta) + eps[a][1] * hess(i, 1, beta);
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_CURL_GRADIENT_HPP
