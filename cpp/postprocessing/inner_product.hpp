#ifndef PYCK_INNER_PRODUCT_HPP
#define PYCK_INNER_PRODUCT_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "../basis/tensor_product.hpp"
#include "../elements/element_values.hpp"
#include "../geometry/patch.hpp"
#include "../quadrature/quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Parametric quadrature points and physical integration measures.
 *
 * For each non-degenerate knot span of @p patch, maps the reference quadrature
 * to the span's parametric box and computes dΩ = w_q · |J(q)|, where |J| is
 * obtained from the patch's intrinsic geometry.
 *
 * @param patch       Patch defining the integration domain.
 * @param quadrature  Reference quadrature rule (mapped per element).
 * @return A pair `(params, dV)`:
 *           - `params` of shape `(Q_total, d)`: parametric coordinates per qpt,
 *           - `dV`    of length `Q_total`:     w_q · |J(q)|.
 *         Both share the same flat element-major ordering.
 */
template <std::floating_point T, std::size_t d>
std::pair<ColMatrix<T, d>, Vector<T>>
eval_quadrature_data(const Patch<T, d>& patch,
                     const QuadratureRule<T, d>& quadrature)
{
    const Index Q = static_cast<Index>(quadrature.num_points());

    ElementValues<T, d> ev(patch, Index(1), Flags::Deriv1, quadrature);
    const std::size_t num_live = static_cast<std::size_t>(ev.num_elements());

    ColMatrix<T, d> params_out(num_live * Q, static_cast<Index>(d));
    Vector<T>       dV_out(num_live * Q);
    Index out = 0;

    for (std::size_t e = 0; e < num_live; ++e)
    {
        ev.reinit(e);

        params_out.block(out, 0, Q, static_cast<Index>(d)) = ev.mapped_pts_;
        for (Index q = 0; q < Q; ++q)
            dV_out(out + q) = ev.mapped_weights_(q) * ev.jac(q);

        out += Q;
    }

    params_out.conservativeResize(out, Eigen::NoChange);
    dV_out.conservativeResize(out);
    return {std::move(params_out), std::move(dV_out)};
}

/**
 * @brief L² inner product ⟨f, g⟩ = ∫_Ω (f · g) dΩ on a patch.
 *
 * Generates quadrature points + integration measures from @p patch and
 * @p quadrature, then accumulates Σ_q (Σ_c f_vals(q, c) · g_vals(q, c)) · dV(q).
 * Both value matrices must be pre-evaluated at the quadrature points returned
 * by @ref eval_quadrature_data (caller's responsibility).
 *
 * @param patch       Patch defining the integration domain.
 * @param quadrature  Reference quadrature rule (mapped per element).
 * @param f_vals      (Q_total, k) field f sampled at the quadrature points.
 * @param g_vals      (Q_total, k) field g sampled at the quadrature points.
 * @return Scalar inner product.
 */
template <std::floating_point T, std::size_t d>
T inner_product(const Patch<T, d>& patch,
                const QuadratureRule<T, d>& quadrature,
                const Matrix<T>& f_vals,
                const Matrix<T>& g_vals)
{
    if (f_vals.rows() != g_vals.rows() || f_vals.cols() != g_vals.cols()) {
        throw std::runtime_error(
            "inner_product: f and g must have matching shape; got (" +
            std::to_string(f_vals.rows()) + ", " + std::to_string(f_vals.cols()) +
            ") vs (" +
            std::to_string(g_vals.rows()) + ", " + std::to_string(g_vals.cols()) + ").");
    }

    auto [params, dV] = eval_quadrature_data(patch, quadrature);

    if (f_vals.rows() != dV.size()) {
        throw std::runtime_error(
            "inner_product: f has " + std::to_string(f_vals.rows()) +
            " rows but the patch+quadrature produced " + std::to_string(dV.size()) +
            " integration points. Evaluate f and g on eval_quadrature_data(patch, "
            "quadrature) first.");
    }

    return (f_vals.array() * g_vals.array()).rowwise().sum().matrix().dot(dV);
}

} // namespace pyck

#endif // PYCK_INNER_PRODUCT_HPP
