#ifndef PYCK_QUADRATURE_UTILS_HPP
#define PYCK_QUADRATURE_UTILS_HPP

#include <cmath>
#include <array>

#include "../geometry/patch.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Compute the physical x-coordinates of all active quadrature points
 *        across a 1-D patch.
 *
 * Iterates over every knot span of the patch, skips zero-volume (clamped)
 * spans, maps the reference quadrature points to the parametric domain via
 * QuadratureRule::map_to_domain, and evaluates the physical geometry via
 * Patch::eval_geometry.  The result is the flat concatenation of x-coordinates
 * in the same element order that LoadCondition expects.
 *
 * @tparam T  Scalar type.
 * @param patch       The 1-D parametric patch.
 * @param quadrature  Quadrature rule (same instance used by LoadCondition).
 * @return Flat vector of physical x-coordinates, length = n_active_elements × Q.
 */
template <std::floating_point T>
Vector<T> eval_physical_quadrature_points(
    const Patch<T, 1>& patch,
    const QuadratureRule<T>& quadrature)
{
    const Index Q          = static_cast<Index>(quadrature.num_points());
    const Index num_spans  = static_cast<Index>(patch.basis(0).knot_vector().num_spans());

    Vector<T> result(num_spans * Q);
    Index out = 0;

    for (Index s = 0; s < num_spans; ++s)
    {
        auto [lo, hi] = patch.basis(0).knot_vector().span_bounds(s);
        if (std::abs(hi - lo) < static_cast<T>(1e-14))
            continue;  // skip zero-volume clamped span

        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);

        std::array<Index, 1> span_arr = {s};
        auto geom = patch.eval_geometry(mapped_pts, span_arr, 0);
        result.segment(out, Q) = geom[0].col(0);
        out += Q;
    }

    result.conservativeResize(out);
    return result;
}

} // namespace pyck

#endif // PYCK_QUADRATURE_UTILS_HPP
