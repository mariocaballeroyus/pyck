#ifndef PYCK_EVAL_GLOBAL_SHAPE_HPP
#define PYCK_EVAL_GLOBAL_SHAPE_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "../elements/element.hpp"
#include "../geometry/intrinsic_geometry.hpp"
#include "../geometry/patch.hpp"
#include "../basis/tensor_product.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Evaluate a global shape matrix at a set of global parametric points.
 *
 * Loops over the unique knot spans implied by `params`, builds the per-point
 * basis derivatives and intrinsic geometry, calls the supplied per-span
 * functor (which returns a reference to the element's shape-matrix
 * workspace), and scatters into a global `(n_pts * n_rows_per_pt, n_cp *
 * ndof)` matrix.
 *
 * @param patch   Patch carrying basis and control points.
 * @param element Element formulation (provides ndof, min_order, shape mats).
 * @param eval_at_span  Callable
 *                      `(elem, patch, basis, ig) -> const Matrix<T>&`
 *                      that triggers the per-span shape matrix computation
 *                      on the element and returns a reference to the
 *                      element-owned workspace holding the result.
 * @param params  `(Q, d)` parametric coordinates (column-major).
 * @return Global shape matrix of shape `(Q * k, n_cp * ndof)`.
 */
template <std::floating_point T, std::size_t d, typename Callable>
Matrix<T> eval_global_shape(const Patch<T, d>& patch,
                            const Element<T, d>& element,
                            Callable&& eval_at_span,
                            const ColMatrix<T, d>& params)
{
    const Index n_pts    = static_cast<Index>(params.rows());
    const Index ndof     = static_cast<Index>(element.num_node_dofs());
    const Index n_global = static_cast<Index>(patch.num_control_pts()) * ndof;
    const std::size_t order = element.min_order();

    std::array<Index, d> n_spans;
    for (std::size_t dir = 0; dir < d; ++dir)
        n_spans[dir] = static_cast<Index>(
            patch.basis(dir).knot_vector().num_spans());

    Matrix<T>           N_global;
    std::vector<Index>  active_cp;         // Reused across spans (lazy-sized).
    Index               n_rows_per_pt = 0;
    bool                initialized = false;

    for (Index i = 0; i < n_pts; ++i)
    {
        ColMatrix<T, d> pt = params.row(i);

        std::array<Index, d> span_idx;
        for (std::size_t dir = 0; dir < d; ++dir)
            span_idx[dir] = patch.basis(dir).find_span(
                static_cast<T>(pt(0, static_cast<Index>(dir))));

        Index flat_span = span_idx[0];
        if constexpr (d >= 2)
            flat_span += span_idx[1] * n_spans[0];

        auto bd      = patch.tensor_product().eval_all(pt, order);
        auto act_pts = patch.active_control_pts(flat_span);
        IntrinsicGeometry<T, d> lf(bd, act_pts, Index(bd.size()) - 1);

        const Matrix<T>& N_span = eval_at_span(element, patch, bd, lf);

        if (!initialized)
        {
            n_rows_per_pt = static_cast<Index>(N_span.rows());
            N_global = Matrix<T>::Zero(n_pts * n_rows_per_pt, n_global);
            initialized = true;
        }

        patch.dof_mapper().get_element_cps(flat_span, active_cp);
        const Index n_active = static_cast<Index>(active_cp.size());

        const Index row_start = i * n_rows_per_pt;
        for (Index k = 0; k < n_active; ++k)
        {
            for (Index v = 0; v < ndof; ++v)
            {
                N_global.block(row_start, active_cp[k] * ndof + v, n_rows_per_pt, 1)
                    .noalias() = N_span.col(k * ndof + v);
            }
        }
    }

    if (!initialized)
        return Matrix<T>(0, n_global);
    return N_global;
}

} // namespace pyck

#endif // PYCK_EVAL_GLOBAL_SHAPE_HPP
