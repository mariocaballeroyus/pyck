#include "christoffels.hpp"

#include <Eigen/Core>

namespace pyck
{

template <std::floating_point T>
ChristoffelSymbols<T, 1>
eval_christoffel(const LocalFrame<T, 1>& local)
{
    const Index Q = local.a1.rows();

    ChristoffelSymbols<T, 1> chr;
    chr.G1_11.resize(Q);

    for (Index q = 0; q < Q; ++q) 
    {
        // Γ¹_{11} = (a_1 · a_{1,1}) / g_{11}
        chr.G1_11(q) = local.a1.row(q).dot(local.a11.row(q)) * local.g_inv_11(q);
    }
    return chr;
}

template <std::floating_point T>
ChristoffelSymbols<T, 2>
eval_christoffel(const LocalFrame<T, 2>& local)
{
    const Index Q = local.a1.rows();

    ChristoffelSymbols<T, 2> chr;
    chr.G1_11.resize(Q); chr.G1_12.resize(Q); chr.G1_22.resize(Q);
    chr.G2_11.resize(Q); chr.G2_12.resize(Q); chr.G2_22.resize(Q);

    for (Index q = 0; q < Q; ++q) 
    {
        // a^δ = g^{δβ} a_β
        const auto a1_q = local.a1.row(q);
        const auto a2_q = local.a2.row(q);
        const T gi11 = local.g_inv(q, 0);
        const T gi12 = local.g_inv(q, 1);
        const T gi22 = local.g_inv(q, 2);
        const Eigen::RowVector<T, 3> aup1 = gi11 * a1_q + gi12 * a2_q;
        const Eigen::RowVector<T, 3> aup2 = gi12 * a1_q + gi22 * a2_q;

        const auto a11_q = local.a11.row(q);
        const auto a12_q = local.a12.row(q);
        const auto a22_q = local.a22.row(q);

        chr.G1_11(q) = aup1.dot(a11_q);
        chr.G1_12(q) = aup1.dot(a12_q);
        chr.G1_22(q) = aup1.dot(a22_q);
        chr.G2_11(q) = aup2.dot(a11_q);
        chr.G2_12(q) = aup2.dot(a12_q);
        chr.G2_22(q) = aup2.dot(a22_q);
    }
    return chr;
}

template <std::floating_point T>
ChristoffelSymbols<T, 3>
eval_christoffel(const LocalFrame<T, 3>& local)
{
    const Index Q = local.a1.rows();

    ChristoffelSymbols<T, 3> chr;
    chr.G1_11.resize(Q); chr.G1_12.resize(Q); chr.G1_13.resize(Q);
    chr.G1_22.resize(Q); chr.G1_23.resize(Q); chr.G1_33.resize(Q);
    chr.G2_11.resize(Q); chr.G2_12.resize(Q); chr.G2_13.resize(Q);
    chr.G2_22.resize(Q); chr.G2_23.resize(Q); chr.G2_33.resize(Q);
    chr.G3_11.resize(Q); chr.G3_12.resize(Q); chr.G3_13.resize(Q);
    chr.G3_22.resize(Q); chr.G3_23.resize(Q); chr.G3_33.resize(Q);

    for (Index q = 0; q < Q; ++q)
    {
        // a^γ = g^{γδ} a_δ;  g_inv layout = (11, 12, 13, 22, 23, 33).
        const auto a1_q = local.a1.row(q);
        const auto a2_q = local.a2.row(q);
        const auto a3_q = local.a3.row(q);
        const T gi11 = local.g_inv(q, 0);
        const T gi12 = local.g_inv(q, 1);
        const T gi13 = local.g_inv(q, 2);
        const T gi22 = local.g_inv(q, 3);
        const T gi23 = local.g_inv(q, 4);
        const T gi33 = local.g_inv(q, 5);
        const Eigen::RowVector<T, 3> aup1 = gi11 * a1_q + gi12 * a2_q + gi13 * a3_q;
        const Eigen::RowVector<T, 3> aup2 = gi12 * a1_q + gi22 * a2_q + gi23 * a3_q;
        const Eigen::RowVector<T, 3> aup3 = gi13 * a1_q + gi23 * a2_q + gi33 * a3_q;

        const auto a11_q = local.a11.row(q);
        const auto a12_q = local.a12.row(q);
        const auto a13_q = local.a13.row(q);
        const auto a22_q = local.a22.row(q);
        const auto a23_q = local.a23.row(q);
        const auto a33_q = local.a33.row(q);

        chr.G1_11(q) = aup1.dot(a11_q); chr.G1_12(q) = aup1.dot(a12_q); chr.G1_13(q) = aup1.dot(a13_q);
        chr.G1_22(q) = aup1.dot(a22_q); chr.G1_23(q) = aup1.dot(a23_q); chr.G1_33(q) = aup1.dot(a33_q);

        chr.G2_11(q) = aup2.dot(a11_q); chr.G2_12(q) = aup2.dot(a12_q); chr.G2_13(q) = aup2.dot(a13_q);
        chr.G2_22(q) = aup2.dot(a22_q); chr.G2_23(q) = aup2.dot(a23_q); chr.G2_33(q) = aup2.dot(a33_q);

        chr.G3_11(q) = aup3.dot(a11_q); chr.G3_12(q) = aup3.dot(a12_q); chr.G3_13(q) = aup3.dot(a13_q);
        chr.G3_22(q) = aup3.dot(a22_q); chr.G3_23(q) = aup3.dot(a23_q); chr.G3_33(q) = aup3.dot(a33_q);
    }
    return chr;
}

// === Template Specializations =======================================================

template ChristoffelSymbols<double, 1> eval_christoffel<double>(const LocalFrame<double, 1>&);
template ChristoffelSymbols<double, 2> eval_christoffel<double>(const LocalFrame<double, 2>&);
template ChristoffelSymbols<double, 3> eval_christoffel<double>(const LocalFrame<double, 3>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template ChristoffelSymbols<float, 1> eval_christoffel<float>(const LocalFrame<float, 1>&);
template ChristoffelSymbols<float, 2> eval_christoffel<float>(const LocalFrame<float, 2>&);
template ChristoffelSymbols<float, 3> eval_christoffel<float>(const LocalFrame<float, 3>&);
#endif

} // namespace pyck
