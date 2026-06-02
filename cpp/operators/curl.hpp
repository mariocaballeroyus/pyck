#ifndef PYCK_CURL_HPP
#define PYCK_CURL_HPP

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
 *        @f$ \varepsilon_\alpha{}^\beta = g_{\alpha\mu}\,\epsilon^{\mu\beta}/\sqrt{g} @f$
 *        (so @f$ \varepsilon_\alpha{}^0 = -g_{\alpha 1}/\sqrt{g} @f$,
 *        @f$ \varepsilon_\alpha{}^1 = g_{\alpha 0}/\sqrt{g} @f$).
 *
 * The divergence-free (rotational) counterpart of @ref CovariantGradient in a
 * surface Helmholtz split. A non-owning view over the basis gradient, the covariant
 * metric, and the Jacobian. Surface operator (d = 2).
 */
template <std::floating_point T, std::size_t d>
struct Curl
{
    const std::vector<Matrix<T>>& results;
    const Matrix<T>&              g_data;
    const Vector<T>&              jac;

    /// @brief @f$ (\mathrm{curl}\,N^i)_\alpha(q) @f$.
    T operator()(Index i, Index alpha, Index q) const
    {
        constexpr Index dd = static_cast<Index>(d);
        const T invJ = T(1) / jac(q);
        const T eps0 = -g_data(q, pack2<d>(alpha, 1)) * invJ;   // ε_α^0
        const T eps1 =  g_data(q, pack2<d>(alpha, 0)) * invJ;   // ε_α^1
        return eps0 * results[1](i * dd + 0, q)
             + eps1 * results[1](i * dd + 1, q);
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_CURL_HPP
