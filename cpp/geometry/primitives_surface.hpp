#ifndef PYCK_PRIMITIVES_SURFACE_HPP
#define PYCK_PRIMITIVES_SURFACE_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

namespace geometry::surface
{

// === Unit Normal ====================================================================

/**
 * @brief Fill the unit normal from the covariant tangent basis and the surface 
 *        Jacobian.
 * 
 * @param pos_derivs Per-order position derivatives sampled at Q quadrature points.
 * @param jacobians  Surface jacobians at each quadrature point.
 * @param normal_out Output array for the unit normals A_3 at each point.
 */
template <std::floating_point T>
inline void compute_normal(const std::vector<ColMatrix<T, 3>>& pos_derivs,
                           const Vector<T>& jacobians,
                           ColMatrix<T, 3>& normal_out)
{
    // Aliases matching differential geometry notation
    const auto& R   = pos_derivs;
    const auto& Jac = jacobians;
    auto& A3        = normal_out;

    // Resize output array to number of quadrature points
    const Index n_gp = Jac.size();
    A3.resize(n_gp, 3);

    auto a_view = [&](Index i) {
        return R[1].middleRows(i * n_gp, n_gp);
    };

    for (Index q = 0; q < n_gp; ++q) {
        // Compute the Jacobian \sqrt{ det A_{αβ} }
        const T J = Jac(q);
        // Compute cross product A_1 × A_2
        const auto A1 = a_view(0).row(q).transpose();
        const auto A2 = a_view(1).row(q).transpose();
        Vector3<T> v = A1.cross(A2);
        // A_3 = (A_1 × A_2) / \sqrt{ det A_{αβ} }
        if (J > T(1e-14)) v /= J;
        else              v = Vector3<T>(T(0), T(0), T(1));
        // Fill output array
        A3.row(q) = v.transpose();
    }
}

// === Curvature (Second Fundamental Form) ============================================

/**
 * @brief Fill the second fundamental form (curvatures) from the 2nd order position 
 *        derivatives and the unit normal.
 * 
 * @details Since B is symmetric, only the upper-triangular entries are stored, 
 *          packed via pack2<2>.
 * 
 * @param pos_derivs    Per-order position derivatives sampled at Q quadrature points.
 * @param normal        Unit surface normal at each sample point, one row per point. 
 * @param curvature_out Output array for the packed components of the second fundamental 
 *                      form B_{αβ} at each point.
 */
template <std::floating_point T>
inline void compute_curvature(const std::vector<ColMatrix<T, 3>>& pos_derivs,
                              const ColMatrix<T, 3>& normal,
                              Matrix<T>& curvature_out)
{
    // Aliases matching differential geometry notation
    const auto& R  = pos_derivs;
    const auto& A3 = normal;
    auto& B        = curvature_out;

    // Resize output array to number of quadrature points
    const Index n_gp = A3.rows();
    B.resize(n_gp, 3);

    auto A_d1 = [&](Index i, Index j) {
        // A_{α,β} = R_{,αβ}
        return R[2].middleRows(pack2<2>(i, j) * n_gp, n_gp);
    };

    for (Index q = 0; q < n_gp; ++q) {
        // B_{αβ} = A_{α,β} · A_3
        B(q, pack2<2>(0, 0)) = A_d1(0, 0).row(q).dot(A3.row(q));
        B(q, pack2<2>(0, 1)) = A_d1(0, 1).row(q).dot(A3.row(q));
        B(q, pack2<2>(1, 1)) = A_d1(1, 1).row(q).dot(A3.row(q));
    }
}

// === Normal Derivative (Weingarten) =================================================

/**
 * @brief Fill the first derivatives of the unit normal A_{3,β} from the Weingarten
 *        relation A_{3,β} = −B^α_β A_α, with the shape operator B^α_β = A^{αγ}B_{γβ}
 *        raised from the *already cached* second fundamental form and inverse metric.
 *        Nothing is recomputed: B_{αβ} and A^{αβ} are read, not rebuilt.
 *
 * @param pos_derivs       Per-order position derivatives at Q quadrature points.
 * @param curvature        Packed second fundamental form B_{αβ} (pack2<2> order).
 * @param metric_inv       Packed inverse metric A^{αβ} (pack2<2> order).
 * @param normal_deriv_out Output A_{3,β}, β-major: A_{3,β} at q in row β·Q + q.
 */
template <std::floating_point T>
inline void compute_normal_derivative(const std::vector<ColMatrix<T, 3>>& pos_derivs,
                                      const Matrix<T>& curvature,
                                      const Matrix<T>& metric_inv,
                                      ColMatrix<T, 3>& normal_deriv_out)
{
    const auto& R   = pos_derivs;
    auto&       dA3 = normal_deriv_out;

    const Index n_gp = curvature.rows();
    dA3.resize(2 * n_gp, 3);

    auto a_view = [&](Index i) {
        return R[1].middleRows(i * n_gp, n_gp);
    };

    for (Index q = 0; q < n_gp; ++q) {
        // Cached second fundamental form B_{αβ} and inverse metric A^{αβ}.
        const T B11 = curvature(q, pack2<2>(0, 0)), B12 = curvature(q, pack2<2>(0, 1)),
                B22 = curvature(q, pack2<2>(1, 1));
        const T gi11 = metric_inv(q, pack2<2>(0, 0)), gi12 = metric_inv(q, pack2<2>(0, 1)),
                gi22 = metric_inv(q, pack2<2>(1, 1));

        // Shape operator B^α_β = A^{αγ} B_{γβ}.
        const T Bmix11 = gi11 * B11 + gi12 * B12;   // B^1_1
        const T Bmix12 = gi11 * B12 + gi12 * B22;   // B^1_2
        const T Bmix21 = gi12 * B11 + gi22 * B12;   // B^2_1
        const T Bmix22 = gi12 * B12 + gi22 * B22;   // B^2_2

        const auto A1 = a_view(0).row(q);
        const auto A2 = a_view(1).row(q);

        // Weingarten: A_{3,β} = −B^α_β A_α.
        dA3.row(0 * n_gp + q) = -(Bmix11 * A1 + Bmix21 * A2);   // A_{3,1}
        dA3.row(1 * n_gp + q) = -(Bmix12 * A1 + Bmix22 * A2);   // A_{3,2}
    }
}

// === Second Normal Derivative =======================================================

/**
 * @brief Fill the second derivatives of the unit normal A_{3,βγ} by differentiating
 *        A_3 = m/J, m = A_1 × A_2, J = ‖m‖, twice — straight from the position
 *        derivatives, with no curvature or connection intermediary. With
 *          m_{,α}  = A_{1,α}×A_2 + A_1×A_{2,α},
 *          m_{,αβ} = A_{1,αβ}×A_2 + A_{1,α}×A_{2,β} + A_{1,β}×A_{2,α} + A_1×A_{2,αβ},
 *          J_{,α}  = (m·m_{,α})/J,
 *          J_{,αβ} = (m_{,α}·m_{,β} + m·m_{,αβ})/J − J_{,α}J_{,β}/J,
 *        the quotient rule gives
 *          A_{3,αβ} = m_{,αβ}/J − (m_{,α}J_{,β} + m_{,β}J_{,α} + m J_{,αβ})/J²
 *                     + 2 m J_{,α}J_{,β}/J³.
 *        Reads position derivatives up to order 3 (A_{α,βγ}); the cached surface
 *        Jacobian supplies J.
 *
 * @param pos_derivs        Per-order position derivatives at Q quadrature points.
 * @param jacobians         Surface Jacobians √det A_{αβ} at each point.
 * @param normal_deriv2_out Output A_{3,βγ}, pair-major: A_{3,βγ} at q in row pack2(β,γ)·Q + q.
 */
template <std::floating_point T>
inline void compute_normal_derivative2(const std::vector<ColMatrix<T, 3>>& pos_derivs,
                                       const Vector<T>& jacobians,
                                       ColMatrix<T, 3>& normal_deriv2_out)
{
    const auto& R   = pos_derivs;
    const auto& Jac = jacobians;
    auto&       ddA3 = normal_deriv2_out;

    const Index n_gp = Jac.size();
    ddA3.resize(3 * n_gp, 3);

    // A_α, A_{α,β}, A_{α,βγ} at point q (α, β, γ ∈ {0, 1}).
    auto A    = [&](Index a, Index q) -> Vector3<T> {
        return R[1].row(a * n_gp + q).transpose();
    };
    auto A_d  = [&](Index a, Index b, Index q) -> Vector3<T> {
        return R[2].row(pack2<2>(a, b) * n_gp + q).transpose();
    };
    auto A_dd = [&](Index a, Index b, Index c, Index q) -> Vector3<T> {
        return R[3].row(pack3<2>(a, b, c) * n_gp + q).transpose();
    };

    for (Index q = 0; q < n_gp; ++q) {
        const T J  = Jac(q);
        const T J2 = J * J, J3 = J2 * J;

        const Vector3<T> A1 = A(0, q), A2 = A(1, q);
        const Vector3<T> m  = A1.cross(A2);

        // m_{,α} and J_{,α}.
        Vector3<T> m_d[2];
        T          J_d[2];
        for (Index a = 0; a < 2; ++a) {
            m_d[a] = A_d(0, a, q).cross(A2) + A1.cross(A_d(1, a, q));
            J_d[a] = m.dot(m_d[a]) / J;
        }

        // A_{3,αβ} for the unique pairs (0,0), (0,1), (1,1).
        for (Index a = 0; a < 2; ++a)
            for (Index b = a; b < 2; ++b) {
                const Vector3<T> m_dd = A_dd(0, a, b, q).cross(A2)
                                      + A_d(0, a, q).cross(A_d(1, b, q))
                                      + A_d(0, b, q).cross(A_d(1, a, q))
                                      + A1.cross(A_dd(1, a, b, q));

                const T J_dd = (m_d[a].dot(m_d[b]) + m.dot(m_dd)) / J
                             - J_d[a] * J_d[b] / J;

                const Vector3<T> ddA3_ab =
                      m_dd / J
                    - (m_d[a] * J_d[b] + m_d[b] * J_d[a] + J_dd * m) / J2
                    + (T(2) * J_d[a] * J_d[b] / J3) * m;

                ddA3.row(pack2<2>(a, b) * n_gp + q) = ddA3_ab.transpose();
            }
    }
}

} // namespace geometry::surface

} // namespace pyck

#endif // PYCK_PRIMITIVES_SURFACE_HPP