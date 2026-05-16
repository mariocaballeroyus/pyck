#ifndef PYCK_LOCAL_FRAME_HPP
#define PYCK_LOCAL_FRAME_HPP

#include <cstddef>
#include <concepts>

#include "../types.hpp"
#include "basis_derivs.hpp"

namespace pyck
{

/**
 * @brief Per-quadrature-point local geometric frame.
 */
template <std::floating_point T, std::size_t d>
struct LocalFrame;

/**
 * @brief 1D local geometric frame.
 */
template <std::floating_point T>
struct LocalFrame<T, 1>
{
    ColMatrix<T, 3> a1;              ///< tangent a_1                             (Q×3)
    ColMatrix<T, 3> a11;             ///< tangent derivative a_{1,1}  (order ≥ 2) (Q×3)
    ColMatrix<T, 3> a111;            ///< 2nd tangent deriv a_{1,11}  (order ≥ 3) (Q×3)
    Vector<T>       g11;             ///< metric g_{11}                           (Q)
    Vector<T>       g_inv_11;        ///< inverse metric g^{11}                   (Q)
    Vector<T>       jac;             ///< \sqrt{g_{11}}                           (Q)
};

/**
 * @brief 2D local geometric frame.
 */
template <std::floating_point T>
struct LocalFrame<T, 2>
{
    ColMatrix<T, 3> a1, a2;                  ///< covariant tangents a_α          (Q×3)
    ColMatrix<T, 3> a11, a12, a22;           ///< 1st derivs a_{αβ}    (order ≥ 2) (Q×3)
    ColMatrix<T, 3> a111, a112, a122, a222;  ///< 2nd derivs a_{αβγ}   (order ≥ 3) (Q×3)
    ColMatrix<T, 3> g;                       ///< metric (g_11, g_12, g_22)       (Q×3)
    ColMatrix<T, 3> g_inv;                   ///< inverse metric (g^11, g^12, g^22)(Q×3)
    Vector<T>       jac;                     ///< √det g                          (Q)
};

/**
 * @brief 3D local geometric frame (volume in 3D physical space).
 *
 * The 6-column metric layout uses the upper-triangular ordering
 * (g_11, g_12, g_13, g_22, g_23, g_33); g_inv follows the same convention.
 */
template <std::floating_point T>
struct LocalFrame<T, 3>
{
    ColMatrix<T, 3> a1, a2, a3;                              ///< covariant tangents a_α          (Q×3)
    ColMatrix<T, 3> a11, a12, a13, a22, a23, a33;            ///< 1st derivs a_{αβ}    (order ≥ 2) (Q×3)
    ColMatrix<T, 3> a111, a112, a113, a122, a123, a133,
                    a222, a223, a233, a333;                  ///< 2nd derivs a_{αβγ}   (order ≥ 3) (Q×3)
    ColMatrix<T, 6> g;                                       ///< metric, upper-tri ordering      (Q×6)
    ColMatrix<T, 6> g_inv;                                   ///< inverse metric, upper-tri       (Q×6)
    Vector<T>       jac;                                     ///< √det g                          (Q)
};

/**
 * @brief Evaluate the local geometric frame at the given coordinates in the
 *        parametric domain.
 *
 * @param basis    The basis functions and their derivatives.
 * @param act_pts  The active control points.
 * @return A struct containing the local geometric frame.
 */
template <std::floating_point T>
LocalFrame<T, 1>
eval_local_frame(const BasisDerivs<T, 1>& basis,
                 const ColMatrix<T, 3>& act_pts);

template <std::floating_point T>
LocalFrame<T, 2>
eval_local_frame(const BasisDerivs<T, 2>& basis,
                 const ColMatrix<T, 3>& act_pts);

template <std::floating_point T>
LocalFrame<T, 3>
eval_local_frame(const BasisDerivs<T, 3>& basis,
                 const ColMatrix<T, 3>& act_pts);

} // namespace pyck

#endif // PYCK_LOCAL_FRAME_HPP
