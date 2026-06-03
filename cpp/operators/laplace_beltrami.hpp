#ifndef PYCK_LAPLACE_BELTRAMI_HPP
#define PYCK_LAPLACE_BELTRAMI_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "covariant_hessian.hpp"
#include "../types.hpp"

namespace pyck
{

namespace operators
{

/**
 * @brief Laplace–Beltrami operator @f$ L_i = g^{\alpha\beta} H_{i\alpha\beta} @f$
 *        — the inverse-metric contraction of the covariant Hessian (off-diagonal
 *        Voigt components doubled). Owning: holds a @ref CovariantHessian (its
 *        connection hoisted in the ctor) plus the Voigt-weighted @f$ g^{\alpha\beta} @f$
 *        weights, both formed once for the fixed qp @p q.
 */
template <std::floating_point T, std::size_t d>
struct LaplaceBeltrami
{
    static constexpr std::size_t n_d2 = d * (d + 1) / 2;

    CovariantHessian<T, d> hess;
    std::array<T, n_d2>    w;     ///< Voigt-weighted g^{αβ}: w[p] = (p<d?1:2)·g^{p}.

    /// @brief Build for quadrature point @p q.
    LaplaceBeltrami(const std::vector<Matrix<T>>&       results,
                    const std::vector<ColMatrix<T, 3>>& position_data,
                    const Matrix<T>&                    g_inv_data,
                    Index                               q)
        : hess(results, position_data, g_inv_data, q)
    {
        for (std::size_t p = 0; p < n_d2; ++p)
            w[p] = ((p < d) ? T(1) : T(2)) * g_inv_data(q, static_cast<Index>(p));
    }

    /// @brief Laplace–Beltrami @f$ L_i @f$ of basis function i.
    T operator()(Index i) const
    {
        T l = T(0);
        for (std::size_t p = 0; p < n_d2; ++p)
            l += w[p] * hess.packed(i, static_cast<Index>(p));
        return l;
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_HPP
