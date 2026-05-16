#ifndef PYCK_TESTS_SUPPORT_FACTORIES_HPP
#define PYCK_TESTS_SUPPORT_FACTORIES_HPP

#include <concepts>

#include "basis.hpp"
#include "patch.hpp"

namespace pyck
{

/**
 * @brief Create a straight line segment C(u) = (Lu, 0, 0).
 *        Control points are placed at Greville abscissae so that the linear
 *        mapping is represented exactly by any polynomial degree.
 *
 * @param basis B-spline basis in the u direction.
 * @param length Physical length of the segment along the x-axis.
 * @return A Patch<T, 1> representing the line segment along the x-axis.
 */
template <std::floating_point T>
Patch<T, 1> line_segment(Ptr<const Basis<T>> basis, T length);

/**
 * @brief Create a flat rectangular surface patch in the xy-plane.
 *
 * Control points are placed at the tensor-product Greville abscissae
 * so that the bilinear map x(u,v) = (width·u, height·v, 0) is exact
 * for any polynomial degree.
 *
 * @param basis_u  Basis in the u direction.
 * @param basis_v  Basis in the v direction.
 * @param width    Physical width  (x-extent).
 * @param height  Physical height of the rectangle along the y-axis.
 * @return A Patch<T, 2> representing the flat surface.
 */
template <std::floating_point T>
Patch<T, 2> rectangle(Ptr<const Basis<T>> basis_u,
                      Ptr<const Basis<T>> basis_v,
                      T width,
                      T height);

/**
 * @brief Create a flat axis-aligned box patch [0, L_x] × [0, L_y] × [0, L_z].
 *
 * Control points are placed at the tensor-product Greville abscissae so that
 * the trilinear map x(u,v,w) = (L_x·u, L_y·v, L_z·w) is exact for any
 * polynomial degree.
 *
 * @param basis_u  Basis in the u direction.
 * @param basis_v  Basis in the v direction.
 * @param basis_w  Basis in the w direction.
 * @param length_x Physical extent along x.
 * @param length_y Physical extent along y.
 * @param length_z Physical extent along z.
 * @return A Patch<T, 3> representing the box.
 */
template <std::floating_point T>
Patch<T, 3> box(Ptr<const Basis<T>> basis_u,
                Ptr<const Basis<T>> basis_v,
                Ptr<const Basis<T>> basis_w,
                T length_x,
                T length_y,
                T length_z);

} // namespace pyck

#endif // PYCK_TESTS_SUPPORT_FACTORIES_HPP
