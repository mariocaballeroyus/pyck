#ifndef PYCK_TEST_SUPPORT_ELEMENT_VALUES_AT_HPP
#define PYCK_TEST_SUPPORT_ELEMENT_VALUES_AT_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "../../cpp/elements/element_values.hpp"
#include "../../cpp/geometry/patch.hpp"
#include "../../cpp/types.hpp"

namespace pyck::test
{

/**
 * @brief Test helper: build and reinit an `ElementValues` workspace at a
 *        fixed set of parametric points. Used by tests that hand-pick points
 *        rather than iterating elements through their own quadrature. All
 *        rows of @p pts must lie in the same span on every direction (uses
 *        row 0 to find the spans).
 */
template <std::floating_point T, std::size_t d, typename PtsLike>
inline ElementValues<T, d>
element_values_at(const Patch<T, d>& patch,
                  const PtsLike& pts,
                  Index basis_order, unsigned flags)
{
    ElementValues<T, d> ev(patch, basis_order, flags,
                           static_cast<std::size_t>(pts.rows()));
    std::array<Index, d> spans;
    for (std::size_t i = 0; i < d; ++i)
        spans[i] = patch.basis(i).find_span(
            static_cast<T>(pts(0, static_cast<Index>(i))));
    ColMatrix<T, d> pts_copy = pts;
    ev.reinit_on_pts(spans, pts_copy);
    return ev;
}

} // namespace pyck::test

#endif // PYCK_TEST_SUPPORT_ELEMENT_VALUES_AT_HPP
