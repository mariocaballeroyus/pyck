#ifndef PYCK_COVARIANT_GRADIENT_HPP
#define PYCK_COVARIANT_GRADIENT_HPP

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
 * @brief Covariant gradient of a contravariant in-plane vector field:
 *        @f$ D_{i\lambda\alpha\beta} = A_{\alpha\lambda} N^i_{,\beta}
 *            + (\mathbf{A}_\alpha\cdot\mathbf{A}_{\lambda,\beta}) N^i @f$,
 *        so that @f$ u_{\alpha|\beta} = D_{i\lambda\alpha\beta}\, u^{\lambda}_i @f$.
 *
 * Owning operator: the constructor forms the per-quadrature-point connectivity — the
 * covariant metric @f$ g_{\alpha\lambda} @f$ and the first-kind connection
 * @f$ \mathbf{A}_\alpha\cdot\mathbf{A}_{\lambda,\beta} @f$ (both plain base-vector dot
 * products, no inverse metric) — once for the fixed qp @p q, so each call is a pure
 * contraction. Build once per qp on the stack, evaluate per basis function.
 */
template <std::floating_point T, std::size_t d>
struct CovariantGradient
{
    static constexpr std::size_t n_d2 = d * (d + 1) / 2;

    const std::vector<Matrix<T>>&      results;
    Index                              q;
    std::array<T, n_d2>                g;       ///< g_{αλ}, Voigt-packed.
    std::array<std::array<T, n_d2>, d> conn;    ///< conn[α][p] = A_α·A_{,p}.

    /// @brief Build for quadrature point @p q, forming the metric and first-kind
    ///        connection once from base vectors.
    CovariantGradient(const std::vector<Matrix<T>>&       results,
                      const std::vector<ColMatrix<T, 3>>& position_data,
                      Index                               q)
        : results(results), q(q)
    {
        const Index Q = position_data[0].rows();
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t l = a; l < d; ++l)
                g[pack2<d>(a, l)] = position_data[1].row(static_cast<Index>(a) * Q + q)
                                      .dot(position_data[1].row(static_cast<Index>(l) * Q + q));
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t p = 0; p < n_d2; ++p)
                conn[a][p] = position_data[1].row(static_cast<Index>(a) * Q + q)
                               .dot(position_data[2].row(static_cast<Index>(p) * Q + q));
    }

    /// @brief @f$ D_{i\lambda\alpha\beta} @f$.
    T operator()(Index i, Index lambda, Index alpha, Index beta) const
    {
        constexpr Index dd = static_cast<Index>(d);
        return g[pack2<d>(alpha, lambda)] * results[1](i * dd + beta, q)
             + conn[static_cast<std::size_t>(alpha)][pack2<d>(lambda, beta)] * results[0](i, q);
    }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_COVARIANT_GRADIENT_HPP
