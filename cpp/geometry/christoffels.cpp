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

// === Template Specializations =======================================================

template ChristoffelSymbols<double, 1> eval_christoffel<double>(const LocalFrame<double, 1>&);
template ChristoffelSymbols<double, 2> eval_christoffel<double>(const LocalFrame<double, 2>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template ChristoffelSymbols<float, 1> eval_christoffel<float>(const LocalFrame<float, 1>&);
template ChristoffelSymbols<float, 2> eval_christoffel<float>(const LocalFrame<float, 2>&);
#endif

} // namespace pyck
