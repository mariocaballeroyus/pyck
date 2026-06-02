#ifndef PYCK_CURL_GRADIENT_HPP
#define PYCK_CURL_GRADIENT_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "covariant_hessian.hpp"
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
 *        @ref CovariantHessian and the permutation tensor is covariantly constant
 *        (so @f$ \varepsilon_\alpha{}^\delta\,\psi_{|\delta\beta}
 *           = (\varepsilon_\alpha{}^\delta\,\psi_{|\delta})_{|\beta} @f$).
 *
 * Built on the covariant Hessian exactly as @ref LaplaceBeltrami is. Returns the
 * (non-symmetric) (α, β) component; consumers symmetrise as the strain measure
 * requires. Surface operator (d = 2).
 */
template <std::floating_point T, std::size_t d>
struct CurlGradient
{
    const std::vector<Matrix<T>>&       results;
    const std::vector<ColMatrix<T, 3>>& position_data;
    const Matrix<T>&                    g_data;
    const Matrix<T>&                    g_inv_data;
    const Vector<T>&                    jac;

    /// @brief @f$ \varepsilon_\alpha{}^\delta H_{i\delta\beta}(q) @f$.
    T operator()(Index i, Index alpha, Index beta, Index q) const
    {
        const CovariantHessian<T, d> hess{results, position_data, g_inv_data};
        const T invJ = T(1) / jac(q);
        const T eps0 = -g_data(q, pack2<d>(alpha, 1)) * invJ;   // ε_α^0
        const T eps1 =  g_data(q, pack2<d>(alpha, 0)) * invJ;   // ε_α^1
        return eps0 * hess(i, 0, beta, q) + eps1 * hess(i, 1, beta, q);
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_CURL_GRADIENT_HPP
