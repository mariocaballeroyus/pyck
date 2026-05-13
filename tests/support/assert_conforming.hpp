#ifndef PYCK_TESTS_SUPPORT_ASSERT_CONFORMING_HPP
#define PYCK_TESTS_SUPPORT_ASSERT_CONFORMING_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <concepts>

#include "patch_boundary.hpp"

namespace pyck
{

/**
 * @brief Verify that two boundaries on either side of an interface are
 *        conforming: same polynomial degree and number of basis functions,
 *        identical (possibly reversed) knot vector along the free direction,
 *        and physically coincident control points.
 *
 *        Throws `std::invalid_argument` on the first mismatch encountered.
 *
 * @param side_a   Boundary on side A of the interface.
 * @param side_b   Boundary on side B of the interface.
 * @param reverse  If true, side_b's free-direction parametrisation is the
 *                 reverse of side_a's.
 * @param tol      Absolute tolerance for knot and control-point comparison.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
void assert_conforming(const PatchBoundary<T, d>& side_a,
                       const PatchBoundary<T, d>& side_b,
                       bool reverse = false,
                       T tol = T(1e-12))
{
    const auto& basis_a = side_a.basis(0);
    const auto& basis_b = side_b.basis(0);

    if (basis_a.degree() != basis_b.degree()) {
        throw std::invalid_argument(
            "assert_conforming: degree mismatch on free direction (a="
            + std::to_string(basis_a.degree()) + ", b="
            + std::to_string(basis_b.degree()) + ").");
    }

    const auto& kv_a = basis_a.knot_vector();
    const auto& kv_b = basis_b.knot_vector();

    if (kv_a.size() != kv_b.size()) {
        throw std::invalid_argument(
            "assert_conforming: knot-vector size mismatch (a="
            + std::to_string(kv_a.size()) + ", b="
            + std::to_string(kv_b.size()) + ").");
    }

    const Index N = kv_a.size();

    // Knot comparison: identical when reverse=false; flipped about the midpoint
    // of side_b's parametric domain when reverse=true.
    const T b_mid = kv_b[0] + kv_b[N - 1];
    for (Index i = 0; i < N; ++i) {
        const T expected_b = reverse ? (b_mid - kv_a[N - 1 - i]) : kv_a[i];
        if (std::abs(kv_b[i] - expected_b) > tol) {
            throw std::invalid_argument(
                "assert_conforming: knot " + std::to_string(i)
                + " mismatch (a=" + std::to_string(kv_a[i])
                + ", b=" + std::to_string(kv_b[i])
                + ", expected_b=" + std::to_string(expected_b)
                + ", reverse=" + (reverse ? "true" : "false") + ").");
        }
    }

    // Control-point coincidence in physical space.
    const auto& cps_a = side_a.control_pts();
    const auto& cps_b = side_b.control_pts();

    if (cps_a.rows() != cps_b.rows()) {
        throw std::invalid_argument(
            "assert_conforming: control-point count mismatch (a="
            + std::to_string(cps_a.rows()) + ", b="
            + std::to_string(cps_b.rows()) + ").");
    }

    const Index Nc = cps_a.rows();
    for (Index i = 0; i < Nc; ++i) {
        const Index j = reverse ? (Nc - 1 - i) : i;
        const T diff = (cps_a.row(i) - cps_b.row(j)).norm();
        if (diff > tol) {
            throw std::invalid_argument(
                "assert_conforming: control point " + std::to_string(i)
                + " on side A does not match control point "
                + std::to_string(j) + " on side B (‖Δ‖="
                + std::to_string(diff) + ", tol="
                + std::to_string(tol) + ").");
        }
    }
}

} // namespace pyck

#endif // PYCK_TESTS_SUPPORT_ASSERT_CONFORMING_HPP
