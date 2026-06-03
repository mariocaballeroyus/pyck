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

} // namespace geometry::surface

} // namespace pyck

#endif // PYCK_PRIMITIVES_SURFACE_HPP