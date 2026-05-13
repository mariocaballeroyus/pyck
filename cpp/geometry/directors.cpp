#include "directors.hpp"

#include <Eigen/Core>

namespace pyck
{

template <std::floating_point T>
ColMatrix<T, 3>
eval_normal(const LocalFrame<T, 2>& local)
{
    const Index Q = local.a1.rows();

    ColMatrix<T, 3> a_3(Q, 3);

    for (Index q = 0; q < Q; ++q) 
    {
        Eigen::Matrix<T, 3, 1> n =
            local.a1.row(q).transpose().cross(local.a2.row(q).transpose());
        const T J = local.jac(q);
        if (J > T(1e-14)) n /= J;
        else n = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
        a_3.row(q) = n.transpose();
    }
    return a_3;
}

template <std::floating_point T>
std::pair<ColMatrix<T, 3>, ColMatrix<T, 3>>
eval_normal_deriv(const LocalFrame<T, 2>& local,
                  const ColMatrix<T, 3>& a_3)
{
    const Index Q = local.a1.rows();

    ColMatrix<T, 3> a_3_1(Q, 3);
    ColMatrix<T, 3> a_3_2(Q, 3);

    for (Index q = 0; q < Q; ++q) 
    {
        const auto a1_q  = local.a1.row(q);
        const auto a2_q  = local.a2.row(q);
        const auto a3_q  = a_3.row(q);
        const auto a11_q = local.a11.row(q);
        const auto a12_q = local.a12.row(q);
        const auto a22_q = local.a22.row(q);
        const T gi11 = local.g_inv(q, 0);
        const T gi12 = local.g_inv(q, 1);
        const T gi22 = local.g_inv(q, 2);

        // Second fundamental form b_{αβ} = a_{αβ}·a_3
        const T b11 = a11_q.dot(a3_q);
        const T b12 = a12_q.dot(a3_q);
        const T b22 = a22_q.dot(a3_q);
        const T bup11 = gi11 * b11 + gi12 * b12;
        const T bup12 = gi11 * b12 + gi12 * b22;
        const T bup21 = gi12 * b11 + gi22 * b12;
        const T bup22 = gi12 * b12 + gi22 * b22;
        a_3_1.row(q) = -(bup11 * a1_q + bup21 * a2_q);
        a_3_2.row(q) = -(bup12 * a1_q + bup22 * a2_q);
    }
    return {std::move(a_3_1), std::move(a_3_2)};
}

// === Template Specializations =======================================================

template ColMatrix<double, 3> eval_normal<double>(const LocalFrame<double, 2>&);
template std::pair<ColMatrix<double, 3>, ColMatrix<double, 3>>
eval_normal_deriv<double>(const LocalFrame<double, 2>&, const ColMatrix<double, 3>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template ColMatrix<float, 3> eval_normal<float>(const LocalFrame<float, 2>&);
template std::pair<ColMatrix<float, 3>, ColMatrix<float, 3>>
eval_normal_deriv<float>(const LocalFrame<float, 2>&, const ColMatrix<float, 3>&);
#endif

} // namespace pyck
