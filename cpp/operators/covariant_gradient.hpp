#ifndef PYCK_COVARIANT_GRADIENT_HPP
#define PYCK_COVARIANT_GRADIENT_HPP

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
 * @brief Covariant gradient of a contravariant in-plane vector field:
 *        @f$ D_{i\lambda\alpha\beta} = A_{\alpha\lambda} N^i_{,\beta}
 *            + (\mathbf{A}_\alpha\cdot\mathbf{A}_{\lambda,\beta}) N^i @f$,
 *        so that @f$ u_{\alpha|\beta} = D_{i\lambda\alpha\beta}\, u^{\lambda}_i @f$.
 *        A view; self-contained from base vectors (covariant metric + first-kind
 *        connection, both plain dot products) — no inverse metric needed.
 */
template <std::floating_point T, std::size_t d>
struct CovariantGradient
{
    const std::vector<Matrix<T>>&       results;
    const std::vector<ColMatrix<T, 3>>& position_data;

    /// @brief @f$ D_{i\lambda\alpha\beta}(q) @f$.
    T operator()(Index i, Index lambda, Index alpha, Index beta, Index q) const
    {
        constexpr Index dd = static_cast<Index>(d);
        const Index Q = results[0].cols();
        const T g_al = position_data[1].row(alpha * Q + q)
                         .dot(position_data[1].row(lambda * Q + q));
        const T conn = position_data[1].row(alpha * Q + q)
                         .dot(position_data[2].row(pack2<d>(lambda, beta) * Q + q));
        return g_al * results[1](i * dd + beta, q) + conn * results[0](i, q);
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_COVARIANT_GRADIENT_HPP
