#ifndef PYCK_LAPLACE_BELTRAMI_HPP
#define PYCK_LAPLACE_BELTRAMI_HPP

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
 *        Voigt components doubled). A view; forms its Hessian on the fly.
 */
template <std::floating_point T, std::size_t d>
struct LaplaceBeltrami
{
    const std::vector<Matrix<T>>&       results;
    const std::vector<ColMatrix<T, 3>>& position_data;
    const Matrix<T>&                    g_inv_data;

    /// @brief Laplace–Beltrami @f$ L_i(q) @f$ of basis function i.
    T operator()(Index i, Index q) const
    {
        constexpr std::size_t n_d2 = d * (d + 1) / 2;
        const CovariantHessian<T, d> hess{results, position_data, g_inv_data};
        T l = T(0);
        for (std::size_t p = 0; p < n_d2; ++p) {
            const T w = (p < d) ? T(1) : T(2);   // Voigt: diagonals first
            l += w * g_inv_data(q, static_cast<Index>(p))
                   * hess.packed(i, static_cast<Index>(p), q);
        }
        return l;
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_HPP
