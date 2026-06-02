#ifndef PYCK_SURFACE_GEOMETRY_HPP
#define PYCK_SURFACE_GEOMETRY_HPP

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

// === Unit normal ====================================================================

/**
 * @brief Fill the unit normal a_3 = (a_1 × a_2) / √det g from position
 *        derivatives (order 1) and the metric Jacobian.
 */
template <std::floating_point T>
inline void compute_normal(const std::vector<ColMatrix<T, 3>>& position_data,
                           const Vector<T>& jac,
                           ColMatrix<T, 3>& n)
{
    const Index Q_ = jac.size();
    n.resize(Q_, 3);

    auto a_view = [&](Index i) {
        return position_data[1].middleRows(i * Q_, Q_);
    };

    for (Index q = 0; q < Q_; ++q) {
        Eigen::Matrix<T, 3, 1> v =
            a_view(0).row(q).transpose().cross(a_view(1).row(q).transpose());
        const T J = jac(q);
        if (J > T(1e-14)) v /= J;
        else              v = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
        n.row(q) = v.transpose();
    }
}

// === Normal derivatives =============================================================

/**
 * @brief Fill ∂_β a_3 from position derivatives (order 2) and the unit normal.
 *        Requires `compute_normal` to have been called.
 */
template <std::floating_point T>
inline void compute_normal_derivatives(const std::vector<ColMatrix<T, 3>>& position_data,
                                       const Vector<T>& jac,
                                       const ColMatrix<T, 3>& n,
                                       ColMatrix<T, 3>& n_d1_data)
{
    const Index Q_ = jac.size();
    n_d1_data.resize(Q_ * 2, 3);

    auto a_view = [&](Index i) {
        return position_data[1].middleRows(i * Q_, Q_);
    };
    auto a_d1_view = [&](Index i, Index j) {
        return position_data[2].middleRows(pack2<2>(i, j) * Q_, Q_);
    };

    for (Index q = 0; q < Q_; ++q)
    {
        const Eigen::Matrix<T, 3, 1> a0  = a_view(0).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a1  = a_view(1).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a3  = n.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a00 = a_d1_view(0, 0).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a01 = a_d1_view(0, 1).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a11 = a_d1_view(1, 1).row(q).transpose();

        const T inv_J = (jac(q) > T(1e-14)) ? T(1) / jac(q) : T(0);

        const Eigen::Matrix<T, 3, 1> dN0 = a00.cross(a1) + a0.cross(a01);
        const Eigen::Matrix<T, 3, 1> dN1 = a01.cross(a1) + a0.cross(a11);

        n_d1_data.middleRows(0 * Q_, Q_).row(q) =
            ((dN0 - a3.dot(dN0) * a3) * inv_J).transpose();
        n_d1_data.middleRows(1 * Q_, Q_).row(q) =
            ((dN1 - a3.dot(dN1) * a3) * inv_J).transpose();
    }
}

// === Curvature ======================================================================

/**
 * @brief Fill the second fundamental form b_{αβ} = a_{αβ} · a_3 from the
 *        order-2 position derivatives and the unit normal. The mixed shape
 *        operator b^α_β = g^{αγ} b_{γβ} is a trivial inverse-metric raise and
 *        is left to consumers that need it.
 */
template <std::floating_point T>
inline void compute_curvature(const std::vector<ColMatrix<T, 3>>& position_data,
                              const ColMatrix<T, 3>& n,
                              Matrix<T>& b_data)
{
    const Index Q_ = n.rows();
    b_data.resize(Q_, 3);

    auto a_d1_view = [&](Index i, Index j) {
        return position_data[2].middleRows(pack2<2>(i, j) * Q_, Q_);
    };

    for (Index q = 0; q < Q_; ++q)
    {
        b_data(q, pack2<2>(0, 0)) = a_d1_view(0, 0).row(q).dot(n.row(q));
        b_data(q, pack2<2>(0, 1)) = a_d1_view(0, 1).row(q).dot(n.row(q));
        b_data(q, pack2<2>(1, 1)) = a_d1_view(1, 1).row(q).dot(n.row(q));
    }
}

} // namespace geometry::surface

} // namespace pyck

#endif // PYCK_SURFACE_GEOMETRY_HPP