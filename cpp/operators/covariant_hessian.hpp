#ifndef PYCK_COVARIANT_HESSIAN_HPP
#define PYCK_COVARIANT_HESSIAN_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Per-(quadrature-point, basis-function) differential-geometry operators.
 *
 * Each operator is an owning view bound to one element's primitive buffers
 * (`{ev.results_, ev.position_data, ev.g_inv_data, …}`): its constructor forms the
 * per-quadrature-point-invariant connectivity (Christoffels, metric-derived terms,
 * permutation factors) once for a fixed qp `q`, storing it in a small fixed-size
 * `std::array` on the stack, and each `operator()` is a pure per-basis-function
 * contraction with no re-derived connection. Construct once per qp inside the
 * element's q-loop and reuse across the basis loop. Composing operators (e.g.
 * @ref LaplaceBeltrami, @ref CurlGradient) hold a @ref CovariantHessian value member
 * so the sub-connection is hoisted once too. Views depend on the buffer types, not on
 * `ElementValues`.
 */
namespace operators
{

/**
 * @brief Covariant Hessian @f$ H_{i\alpha\beta} = N^i_{,\alpha\beta}
 *        - \Gamma^\lambda_{\alpha\beta} N^i_{,\lambda} @f$, with the second-kind
 *        connection @f$ \Gamma^\lambda_{\alpha\beta} = g^{\lambda\varepsilon}
 *        (\mathbf{A}_\varepsilon\cdot\mathbf{A}_{,\alpha\beta}) @f$ hoisted at
 *        construction for the fixed qp @p q. Each call is just the contraction.
 */
template <std::floating_point T, std::size_t d>
struct CovariantHessian
{
    static constexpr std::size_t n_d2 = d * (d + 1) / 2;

    const std::vector<Matrix<T>>&      results;
    Index                              q;
    std::array<std::array<T, n_d2>, d> Gamma;   ///< Γ^λ_{(αβ)=p}, hoisted in the ctor.

    /// @brief Build for quadrature point @p q, forming Γ^λ_{αβ} once from base vectors.
    CovariantHessian(const std::vector<Matrix<T>>&       results,
                     const std::vector<ColMatrix<T, 3>>& position_data,
                     const Matrix<T>&                    g_inv_data,
                     Index                               q)
        : results(results), q(q)
    {
        const Index Q = position_data[0].rows();
        // first[ε][p] = A_ε · A_{,p}, raised by the inverse metric:
        //   Γ^λ_p = Σ_ε g^{λε} first[ε][p].
        std::array<std::array<T, n_d2>, d> first{};
        for (std::size_t e = 0; e < d; ++e)
            for (std::size_t p = 0; p < n_d2; ++p)
                first[e][p] = position_data[1].row(static_cast<Index>(e) * Q + q)
                                .dot(position_data[2].row(static_cast<Index>(p) * Q + q));
        for (std::size_t lam = 0; lam < d; ++lam)
            for (std::size_t p = 0; p < n_d2; ++p) {
                T g = T(0);
                for (std::size_t e = 0; e < d; ++e)
                    g += g_inv_data(q, pack2<d>(lam, e)) * first[e][p];
                Gamma[lam][p] = g;
            }
    }

    /// @brief Hessian for the Voigt-packed pair @p p.
    T packed(Index i, Index p) const
    {
        T h = results[2](i * static_cast<Index>(n_d2) + p, q);
        for (std::size_t lam = 0; lam < d; ++lam)
            h -= Gamma[lam][p]
                   * results[1](i * static_cast<Index>(d) + static_cast<Index>(lam), q);
        return h;
    }

    /// @brief Covariant Hessian @f$ H_{i\alpha\beta} @f$. Symmetric in (α, β).
    T operator()(Index i, Index alpha, Index beta) const
    { return packed(i, pack2<d>(alpha, beta)); }
};

} // namespace operators

} // namespace pyck

#endif // PYCK_COVARIANT_HESSIAN_HPP
