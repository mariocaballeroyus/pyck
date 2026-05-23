#ifndef PYCK_BOUNDARY_GEOMETRY_HPP
#define PYCK_BOUNDARY_GEOMETRY_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "../types.hpp"

namespace pyck
{

namespace geometry::boundary
{

// === Outward normal =================================================================

/**
 * @brief Outward in-surface normal at boundary trace points of a 2D parent:
 *        the direction in the parent's tangent plane perpendicular to the
 *        boundary curve, oriented outward by `sign`.
 *
 * @details The parent surface normal a_3 = (a_1 × a_2) / jac is reconstructed
 *          inline from the parent's order-1 position derivatives and jac, then
 *          n = sign · (t_bdy × a_3) / ‖…‖.
 */
template <std::floating_point T, std::size_t d>
requires (d == 2)
inline void compute_outward_normal(const std::vector<ColMatrix<T, 3>>& boundary_position_data,
                                   const std::vector<ColMatrix<T, 3>>& parent_position_data,
                                   const Vector<T>& parent_jac,
                                   T sign,
                                   ColMatrix<T, 3>& n_out)
{
    const Index Q = parent_jac.size();
    n_out.resize(Q, 3);

    auto par_a = [&](Index i) {
        return parent_position_data[1].middleRows(i * Q, Q);
    };

    for (Index q = 0; q < Q; ++q)
    {
        Eigen::Matrix<T, 3, 1> pa1 = par_a(0).row(q).transpose();
        Eigen::Matrix<T, 3, 1> pa2 = par_a(1).row(q).transpose();
        Eigen::Matrix<T, 3, 1> a3  = pa1.cross(pa2);
        const T jac_p = parent_jac(q);
        if (jac_p > T(1e-14)) a3 /= jac_p;
        else                  a3 = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));

        Eigen::Matrix<T, 3, 1> t = boundary_position_data[1].row(q).transpose();
        Eigen::Matrix<T, 3, 1> n_vec = sign * t.cross(a3);
        const T n_norm = n_vec.norm();
        if (n_norm > T(1e-14)) n_vec /= n_norm;
        n_out.row(q) = n_vec.transpose();
    }
}

/**
 * @brief Outward normal at boundary trace points of a 3D parent: the boundary
 *        surface's own normal, sign-flipped to point out of the parent volume.
 *
 * @details The boundary is a 2D surface in 3D ambient space, so its normal is
 *          n = sign · (a_1^bd × a_2^bd) / ‖…‖, built from the boundary's own
 *          order-1 position derivatives. The parent's geometry is not needed.
 */
template <std::floating_point T, std::size_t d>
requires (d == 3)
inline void compute_outward_normal(const std::vector<ColMatrix<T, 3>>& boundary_position_data,
                                   T sign,
                                   ColMatrix<T, 3>& n_out)
{
    const Index Q = boundary_position_data[1].rows() / 2;
    n_out.resize(Q, 3);

    auto bdy_a = [&](Index i) {
        return boundary_position_data[1].middleRows(i * Q, Q);
    };

    for (Index q = 0; q < Q; ++q)
    {
        Eigen::Matrix<T, 3, 1> a1 = bdy_a(0).row(q).transpose();
        Eigen::Matrix<T, 3, 1> a2 = bdy_a(1).row(q).transpose();
        Eigen::Matrix<T, 3, 1> n_vec = sign * a1.cross(a2);
        const T n_norm = n_vec.norm();
        if (n_norm > T(1e-14)) n_vec /= n_norm;
        n_out.row(q) = n_vec.transpose();
    }
}

// === In-surface tangent =============================================================

/**
 * @brief In-surface unit tangent s = a_3 × n. With a_3 and n both unit and
 *        mutually perpendicular, ‖s‖ = 1 by construction — no normalization.
 *        Only meaningful when the boundary is a curve on a 2D parent (d=2).
 */
template <std::floating_point T>
inline void compute_in_surface_tangent(const ColMatrix<T, 3>& a_3,
                                       const ColMatrix<T, 3>& n,
                                       ColMatrix<T, 3>& s_out)
{
    const Index Q = a_3.rows();
    s_out.resize(Q, 3);
    for (Index q = 0; q < Q; ++q) {
        const Eigen::Matrix<T, 3, 1> a3q = a_3.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> nq  = n.row(q).transpose();
        s_out.row(q) = a3q.cross(nq).transpose();
    }
}

} // namespace geometry::boundary

} // namespace pyck

#endif // PYCK_BOUNDARY_GEOMETRY_HPP
