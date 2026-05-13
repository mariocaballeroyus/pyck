#include "laplace_beltrami.hpp"

namespace pyck
{

template <std::floating_point T>
LaplaceGradAux<T>
compute_laplace_grad_aux(const LocalFrame<T, 2>& local,
                         const ChristoffelSymbols<T, 2>& chr)
{
    const Index Q = local.a1.rows();

    LaplaceGradAux<T> aux;
    aux.G11_d1.resize(Q); aux.G12_d1.resize(Q); aux.G22_d1.resize(Q);
    aux.G11_d2.resize(Q); aux.G12_d2.resize(Q); aux.G22_d2.resize(Q);
    aux.c1.resize(Q);     aux.c2.resize(Q);
    aux.c1_d1.resize(Q);  aux.c2_d1.resize(Q);
    aux.c1_d2.resize(Q);  aux.c2_d2.resize(Q);

    for (Index q = 0; q < Q; ++q) 
    {
        const T G11 = local.g_inv(q, 0);
        const T G12 = local.g_inv(q, 1);
        const T G22 = local.g_inv(q, 2);
        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);

        const auto a1   = local.a1.row(q);
        const auto a2   = local.a2.row(q);
        const auto a11  = local.a11.row(q);
        const auto a12  = local.a12.row(q);
        const auto a22  = local.a22.row(q);
        const auto a111 = local.a111.row(q);
        const auto a112 = local.a112.row(q);
        const auto a122 = local.a122.row(q);
        const auto a222 = local.a222.row(q);

        // Tangent dot products
        const T a1_a11  = a1.dot(a11),  a1_a12  = a1.dot(a12),  a1_a22  = a1.dot(a22);
        const T a2_a11  = a2.dot(a11),  a2_a12  = a2.dot(a12),  a2_a22  = a2.dot(a22);
        const T a11_a11 = a11.squaredNorm(), a11_a12 = a11.dot(a12), a11_a22 = a11.dot(a22);
        const T a12_a12 = a12.squaredNorm(), a12_a22 = a12.dot(a22), a22_a22 = a22.squaredNorm();
        const T a1_a111 = a1.dot(a111), a1_a112 = a1.dot(a112);
        const T a1_a122 = a1.dot(a122), a1_a222 = a1.dot(a222);
        const T a2_a111 = a2.dot(a111), a2_a112 = a2.dot(a112);
        const T a2_a122 = a2.dot(a122), a2_a222 = a2.dot(a222);

        // Metric partials g_{μν,γ} = a_{μγ}·a_ν + a_μ·a_{νγ}
        const T g11_d1 = T(2) * a1_a11;
        const T g11_d2 = T(2) * a1_a12;
        const T g12_d1 = a2_a11 + a1_a12;
        const T g12_d2 = a2_a12 + a1_a22;
        const T g22_d1 = T(2) * a2_a12;
        const T g22_d2 = T(2) * a2_a22;

        // Inverse-metric partials (g^{βγ})_{,α} = -g^{βμ} g^{γν} g_{μν,α}
        const T G11_d1 = -(G11*G11*g11_d1 + T(2)*G11*G12*g12_d1 + G12*G12*g22_d1);
        const T G11_d2 = -(G11*G11*g11_d2 + T(2)*G11*G12*g12_d2 + G12*G12*g22_d2);
        const T G12_d1 = -(G11*G12*g11_d1 + (G11*G22 + G12*G12)*g12_d1 + G12*G22*g22_d1);
        const T G12_d2 = -(G11*G12*g11_d2 + (G11*G22 + G12*G12)*g12_d2 + G12*G22*g22_d2);
        const T G22_d1 = -(G12*G12*g11_d1 + T(2)*G12*G22*g12_d1 + G22*G22*g22_d1);
        const T G22_d2 = -(G12*G12*g11_d2 + T(2)*G12*G22*g12_d2 + G22*G22*g22_d2);

        // Γ^δ_{βγ,α} = (g^{δε})_{,α}(a_ε·a_{βγ}) + g^{δε}(a_{εα}·a_{βγ} + a_ε·a_{βγα})
        const T G1_11_d1 = G11_d1*a1_a11 + G12_d1*a2_a11
                         + G11*(a11_a11 + a1_a111) + G12*(a11_a12 + a2_a111);
        const T G1_12_d1 = G11_d1*a1_a12 + G12_d1*a2_a12
                         + G11*(a11_a12 + a1_a112) + G12*(a12_a12 + a2_a112);
        const T G1_22_d1 = G11_d1*a1_a22 + G12_d1*a2_a22
                         + G11*(a11_a22 + a1_a122) + G12*(a12_a22 + a2_a122);
        const T G2_11_d1 = G12_d1*a1_a11 + G22_d1*a2_a11
                         + G12*(a11_a11 + a1_a111) + G22*(a11_a12 + a2_a111);
        const T G2_12_d1 = G12_d1*a1_a12 + G22_d1*a2_a12
                         + G12*(a11_a12 + a1_a112) + G22*(a12_a12 + a2_a112);
        const T G2_22_d1 = G12_d1*a1_a22 + G22_d1*a2_a22
                         + G12*(a11_a22 + a1_a122) + G22*(a12_a22 + a2_a122);
        const T G1_11_d2 = G11_d2*a1_a11 + G12_d2*a2_a11
                         + G11*(a11_a12 + a1_a112) + G12*(a11_a22 + a2_a112);
        const T G1_12_d2 = G11_d2*a1_a12 + G12_d2*a2_a12
                         + G11*(a12_a12 + a1_a122) + G12*(a12_a22 + a2_a122);
        const T G1_22_d2 = G11_d2*a1_a22 + G12_d2*a2_a22
                         + G11*(a12_a22 + a1_a222) + G12*(a22_a22 + a2_a222);
        const T G2_11_d2 = G12_d2*a1_a11 + G22_d2*a2_a11
                         + G12*(a11_a12 + a1_a112) + G22*(a11_a22 + a2_a112);
        const T G2_12_d2 = G12_d2*a1_a12 + G22_d2*a2_a12
                         + G12*(a12_a12 + a1_a122) + G22*(a12_a22 + a2_a122);
        const T G2_22_d2 = G12_d2*a1_a22 + G22_d2*a2_a22
                         + G12*(a12_a22 + a1_a222) + G22*(a22_a22 + a2_a222);

        // c^δ and (c^δ)_{,α}
        const T c1 = G11*Gam1_11 + T(2)*G12*Gam1_12 + G22*Gam1_22;
        const T c2 = G11*Gam2_11 + T(2)*G12*Gam2_12 + G22*Gam2_22;
        const T c1_d1 = G11_d1*Gam1_11 + T(2)*G12_d1*Gam1_12 + G22_d1*Gam1_22
                      + G11*G1_11_d1 + T(2)*G12*G1_12_d1 + G22*G1_22_d1;
        const T c2_d1 = G11_d1*Gam2_11 + T(2)*G12_d1*Gam2_12 + G22_d1*Gam2_22
                      + G11*G2_11_d1 + T(2)*G12*G2_12_d1 + G22*G2_22_d1;
        const T c1_d2 = G11_d2*Gam1_11 + T(2)*G12_d2*Gam1_12 + G22_d2*Gam1_22
                      + G11*G1_11_d2 + T(2)*G12*G1_12_d2 + G22*G1_22_d2;
        const T c2_d2 = G11_d2*Gam2_11 + T(2)*G12_d2*Gam2_12 + G22_d2*Gam2_22
                      + G11*G2_11_d2 + T(2)*G12*G2_12_d2 + G22*G2_22_d2;

        aux.G11_d1(q) = G11_d1; aux.G12_d1(q) = G12_d1; aux.G22_d1(q) = G22_d1;
        aux.G11_d2(q) = G11_d2; aux.G12_d2(q) = G12_d2; aux.G22_d2(q) = G22_d2;
        aux.c1(q)     = c1;     aux.c2(q)     = c2;
        aux.c1_d1(q)  = c1_d1;  aux.c2_d1(q)  = c2_d1;
        aux.c1_d2(q)  = c1_d2;  aux.c2_d2(q)  = c2_d2;
    }
    return aux;
}

// === Template Specializations =======================================================

template LaplaceGradAux<double> compute_laplace_grad_aux<double>(
    const LocalFrame<double, 2>&, const ChristoffelSymbols<double, 2>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template LaplaceGradAux<float> compute_laplace_grad_aux<float>(
    const LocalFrame<float, 2>&, const ChristoffelSymbols<float, 2>&);
#endif

} // namespace pyck
