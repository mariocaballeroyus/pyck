#include "physical_points.hpp"

#include <array>
#include <cmath>

namespace pyck
{

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> eval_physical_points(
    const Patch<T, d>& patch,
    const QuadratureRule<T, d>& quadrature)
{
    const Index Q = static_cast<Index>(quadrature.num_points());

    // Total number of candidate elements (flat-indexed, u-fastest).
    std::array<std::size_t, d> intervals;
    std::size_t total_elements = 1;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = patch.basis(i).knot_vector().num_spans();
        total_elements *= intervals[i];
    }

    ColMatrix<T, 3> result(total_elements * Q, 3);
    Index out = 0;

    for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx)
    {
        // Decode flat index into per-direction spans (u-fastest).
        std::array<std::size_t, d> spans;
        std::size_t tmp = elem_idx;
        for (std::size_t i = 0; i < d; ++i) {
            spans[i] = tmp % intervals[i];
            tmp /= intervals[i];
        }

        // Skip degenerate spans.
        std::array<T, d> lo, hi;
        bool zero_volume = false;
        for (std::size_t i = 0; i < d; ++i) {
            auto [l, h] = patch.basis(i).knot_vector().span_bounds(spans[i]);
            lo[i] = l;
            hi[i] = h;
            if (std::abs(h - l) < T(1e-14)) {
                zero_volume = true;
                break;
            }
        }
        if (zero_volume) continue;

        // phys_pts = N * act_pts, with N from eval_basis at order 0.
        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        auto basis = eval_basis(patch, mapped_pts, static_cast<Index>(elem_idx), 0);
        auto act_pts = patch.active_control_pts(static_cast<Index>(elem_idx));
        result.block(out, 0, Q, 3).noalias() = basis.N * act_pts;
        out += Q;
    }

    result.conservativeResize(out, Eigen::NoChange);
    return result;
}

// === Template Instantiations ========================================================

template ColMatrix<double, 3> eval_physical_points<double, 1>(
    const Patch<double, 1>&, const QuadratureRule<double, 1>&);
template ColMatrix<double, 3> eval_physical_points<double, 2>(
    const Patch<double, 2>&, const QuadratureRule<double, 2>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template ColMatrix<float, 3> eval_physical_points<float, 1>(
    const Patch<float, 1>&, const QuadratureRule<float, 1>&);
template ColMatrix<float, 3> eval_physical_points<float, 2>(
    const Patch<float, 2>&, const QuadratureRule<float, 2>&);
#endif

} // namespace pyck
