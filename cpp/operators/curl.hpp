#ifndef PYCK_CURL_HPP
#define PYCK_CURL_HPP

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
 * @brief Surface curl (skew gradient) of a scalar basis function:
 *        @f$ (\mathrm{curl}\,N^i)_\alpha = \varepsilon_\alpha{}^\beta N^i_{,\beta} @f$,
 *        with the mixed surface permutation tensor
 *        @f$ \varepsilon_\alpha{}^\beta = A_{\alpha\mu}\,\epsilon^{\mu\beta}/\sqrt{A} @f$
 *        (so @f$ \varepsilon_\alpha{}^0 = -A_{\alpha 1}/\sqrt{A} @f$,
 *        @f$ \varepsilon_\alpha{}^1 = A_{\alpha 0}/\sqrt{A} @f$).
 *
 * The divergence-free (rotational) counterpart of @ref CovariantGradient in a surface
 * Helmholtz split. Owning operator: the constructor forms the mixed permutation once for
 * the fixed qp @p q; each call is a contraction. Surface operator (d = 2).
 */
template <std::floating_point T, std::size_t d>
struct Curl
{
    const std::vector<Matrix<T>>&   basis_derivs;
    Index                           q;
    std::array<std::array<T, d>, d> eps;   ///< ε_α^β.

    /// @brief Build for quadrature point @p q, forming the mixed permutation once.
    Curl(const std::vector<Matrix<T>>& basis_derivs,
         const Matrix<T>& metric, const Vector<T>& jac, Index q)
        : basis_derivs(basis_derivs), q(q)
    {
        const T invJ = T(1) / jac(q);
        for (std::size_t a = 0; a < d; ++a) {
            eps[a][0] = -metric(q, pack2<d>(static_cast<Index>(a), 1)) * invJ;   // ε_α^0
            eps[a][1] =  metric(q, pack2<d>(static_cast<Index>(a), 0)) * invJ;   // ε_α^1
        }
    }

    /// @brief @f$ (\mathrm{curl}\,N^i)_\alpha @f$.
    T operator()(Index i, Index alpha) const
    {
        constexpr Index dd = static_cast<Index>(d);
        const std::size_t a = static_cast<std::size_t>(alpha);
        return eps[a][0] * basis_derivs[1](i * dd + 0, q)
             + eps[a][1] * basis_derivs[1](i * dd + 1, q);
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_CURL_HPP
