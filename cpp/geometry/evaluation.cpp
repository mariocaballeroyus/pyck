#include "evaluation.hpp"
#include <algorithm>

namespace pyck
{

template <std::floating_point T, std::size_t d>
std::vector<Matrix<T>> eval_shape_at(const Patch<T, d>& patch, 
                                     const ColMatrix<T, d>& points, 
                                     std::size_t order)
{
    const Index Q = points.rows();
    const Index order_idx = std::max(Index(1), std::min(static_cast<Index>(order), Index(3)));
    
    // Allocate output structures based on topological dimension
    std::size_t num_result_mats;
    if constexpr (d == 1) {
        num_result_mats = order_idx + 1;
    } else {
        num_result_mats = (order_idx == 1) ? 3 : ((order_idx == 2) ? 6 : 10);
    }

    std::vector<Matrix<T>> result(num_result_mats);
    for (std::size_t k = 0; k < num_result_mats; ++k) {
        result[k] = Matrix<T>::Zero(Q, patch.num_control_pts());
    }

    // Evaluate each point independently
    for (Index q = 0; q < Q; ++q) {
        Index span;
        if constexpr (d == 1) {
            span = patch.basis(0).find_span(points(q, 0));
        } else {
            const Index span_u = patch.basis(0).find_span(points(q, 0));
            const Index span_v = patch.basis(1).find_span(points(q, 1));
            const Index num_intervals_u = patch.basis(0).num_intervals();
            span = span_u + span_v * num_intervals_u;
        }

        ColMatrix<T, d> pt = points.row(q);
        auto [shape_fns_q, jac_q] = patch.eval_shape_functions(pt, span, order);

        // Get active DOFs for this element
        auto active_dofs = patch.dof_mapper().get_element_dofs(span);
        const Index K = static_cast<Index>(active_dofs.size());

        // Copy results to global arrays
        for (std::size_t k = 0; k < num_result_mats; ++k) {
            for (Index loc = 0; loc < K; ++loc) {
                result[k](q, active_dofs[loc]) = shape_fns_q[k](0, loc);
            }
        }
    }

    return result;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> eval_geometry_at(const Patch<T, d>& patch, 
                                 const ColMatrix<T, d>& points)
{
    const Index Q = points.rows();
    ColMatrix<T, 3> result(Q, 3);

    for (Index q = 0; q < Q; ++q) {
        Index span;
        if constexpr (d == 1) {
            span = patch.basis(0).find_span(points(q, 0));
        } else {
            const Index span_u = patch.basis(0).find_span(points(q, 0));
            const Index span_v = patch.basis(1).find_span(points(q, 1));
            const Index num_intervals_u = patch.basis(0).num_intervals();
            span = span_u + span_v * num_intervals_u;
        }

        ColMatrix<T, d> pt = points.row(q);
        result.row(q) = patch.eval_geometry(pt, span).row(0);
    }

    return result;
}

// Explicit instantiations
template std::vector<Matrix<double>> eval_shape_at<double, 1>(const Patch<double, 1>&, const ColMatrix<double, 1>&, std::size_t);
template std::vector<Matrix<double>> eval_shape_at<double, 2>(const Patch<double, 2>&, const ColMatrix<double, 2>&, std::size_t);
template ColMatrix<double, 3> eval_geometry_at<double, 1>(const Patch<double, 1>&, const ColMatrix<double, 1>&);
template ColMatrix<double, 3> eval_geometry_at<double, 2>(const Patch<double, 2>&, const ColMatrix<double, 2>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template std::vector<Matrix<float>> eval_shape_at<float, 1>(const Patch<float, 1>&, const ColMatrix<float, 1>&, std::size_t);
template std::vector<Matrix<float>> eval_shape_at<float, 2>(const Patch<float, 2>&, const ColMatrix<float, 2>&, std::size_t);
template ColMatrix<float, 3> eval_geometry_at<float, 1>(const Patch<float, 1>&, const ColMatrix<float, 1>&);
template ColMatrix<float, 3> eval_geometry_at<float, 2>(const Patch<float, 2>&, const ColMatrix<float, 2>&);
#endif

} // namespace pyck
