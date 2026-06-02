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

/**
 * @brief Test helper: second-kind Christoffel Γ^k_{ij}(q) = g^{kγ}(A_γ·A_{,ij}),
 *        formed from an `ElementValues`' base-vector and inverse-metric accessors.
 *        `ElementValues` stores no connection — consumers build it in place — so
 *        this mirrors that computation, letting the closed-form geometry tests
 *        still assert the connection. Requires `Flags::Deriv2`; symmetric in (i, j).
 */
template <std::floating_point T, std::size_t d>
inline T christoffel2nd(const ElementValues<T, d>& ev,
                        Index k, Index i, Index j, Index q)
{
    T s = T(0);
    for (Index g = 0; g < static_cast<Index>(d); ++g)
        s += ev.g_inv(k, g)(q) * ev.a(g).row(q).dot(ev.a_d1(i, j).row(q));
    return s;
}

} // namespace pyck::test

#endif // PYCK_TEST_SUPPORT_ELEMENT_VALUES_AT_HPP
