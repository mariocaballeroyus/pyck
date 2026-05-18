#include "laplace_beltrami.hpp"

#include "intrinsic_geometry.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace pyck
{

namespace {

/// Symmetric 2-tuple swap to upper-tri form (i ≤ j).
constexpr std::pair<std::size_t, std::size_t>
sym2(std::size_t i, std::size_t j)
{
    return i <= j ? std::pair{i, j} : std::pair{j, i};
}

/// Symmetric 3-tuple sort to upper-tri form (i ≤ j ≤ k) via a 3-element
/// sorting network.
constexpr std::array<std::size_t, 3>
sym3(std::size_t i, std::size_t j, std::size_t k)
{
    std::array<std::size_t, 3> v{i, j, k};
    if (v[0] > v[1]) std::swap(v[0], v[1]);
    if (v[1] > v[2]) std::swap(v[1], v[2]);
    if (v[0] > v[1]) std::swap(v[0], v[1]);
    return v;
}

}  // namespace

template <std::floating_point T, std::size_t d>
LaplaceGradAux<T, d>
compute_laplace_grad_aux(const IntrinsicGeometry<T, d>& ig)
{
    const Index Q = ig.a(0).rows();


    LaplaceGradAux<T, d> aux;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = i; j < d; ++j)
            for (std::size_t a = 0; a < d; ++a)
                aux.G_inv_d[i][j][a].resize(Q);
    for (std::size_t delta = 0; delta < d; ++delta) {
        aux.c[delta].resize(Q);
        for (std::size_t a = 0; a < d; ++a)
            aux.c_d[delta][a].resize(Q);
    }

    for (Index q = 0; q < Q; ++q)
    {
        // Symmetric accessors hiding the upper-tri storage convention.
        auto a1  = [&](std::size_t i) {
            return ig.a(i).row(q);
        };
        auto a2  = [&](std::size_t i, std::size_t j) {
            auto [lo, hi] = sym2(i, j);
            return ig.a_d1(lo, hi).row(q);
        };
        auto a3  = [&](std::size_t i, std::size_t j, std::size_t k) {
            auto v = sym3(i, j, k);
            return ig.a_d2(v[0], v[1], v[2]).row(q);
        };
        auto G   = [&](std::size_t i, std::size_t j) {
            auto [lo, hi] = sym2(i, j);
            return ig.g_inv(lo, hi)(q);
        };
        auto Gam = [&](std::size_t e, std::size_t i, std::size_t j) {
            auto [lo, hi] = sym2(i, j);
            return ig.chr.Gamma(e, lo, hi)(q);
        };

        // ---- Precompute unique dot products once per q ---------------------

        // dot1[μ][i][j] = a_μ · a_{ij}, symmetric in (i, j).
        std::array<std::array<std::array<T, d>, d>, d> dot1{};
        for (std::size_t mu = 0; mu < d; ++mu)
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = i; j < d; ++j) {
                    const T v = a1(mu).dot(a2(i, j));
                    dot1[mu][i][j] = v;
                    dot1[mu][j][i] = v;
                }

        // dot2[i][j][k][l] = a_{ij} · a_{kl}, mirrored on (i, j) and (k, l).
        std::array<std::array<std::array<std::array<T, d>, d>, d>, d> dot2{};
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j)
                for (std::size_t k = 0; k < d; ++k)
                    for (std::size_t l = k; l < d; ++l) {
                        const T v = a2(i, j).dot(a2(k, l));
                        dot2[i][j][k][l] = v;
                        dot2[j][i][k][l] = v;
                        dot2[i][j][l][k] = v;
                        dot2[j][i][l][k] = v;
                    }

        // dot3[μ][i][j][k] = a_μ · a_{ijk}, fully symmetric in (i, j, k).
        std::array<std::array<std::array<std::array<T, d>, d>, d>, d> dot3{};
        for (std::size_t mu = 0; mu < d; ++mu)
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = i; j < d; ++j)
                    for (std::size_t k = j; k < d; ++k) {
                        const T v = a1(mu).dot(a3(i, j, k));
                        std::array<std::size_t, 3> p{i, j, k};
                        do { dot3[mu][p[0]][p[1]][p[2]] = v; }
                        while (std::next_permutation(p.begin(), p.end()));
                    }

        // ---- Scalar arithmetic on the tables -------------------------------

        // g_{ij,α} = a_{iα}·a_j + a_i·a_{jα}, sym in (i, j).
        auto g_d = [&](std::size_t i, std::size_t j, std::size_t alpha) -> T {
            return dot1[j][i][alpha] + dot1[i][j][alpha];
        };

        // (g^{ij})_{,α} = -g^{im} g^{jn} g_{mn,α}, sym in (i, j).
        std::array<std::array<std::array<T, d>, d>, d> Gup_d{};
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j)
                for (std::size_t alpha = 0; alpha < d; ++alpha) {
                    T s = T(0);
                    for (std::size_t m = 0; m < d; ++m)
                        for (std::size_t n = 0; n < d; ++n)
                            s += G(i, m) * G(j, n) * g_d(m, n, alpha);
                    Gup_d[i][j][alpha] = -s;
                    Gup_d[j][i][alpha] = -s;
                    aux.G_inv_d[std::min(i, j)][std::max(i, j)][alpha](q) = -s;
                }

        // Γ^δ_{ij,α} = (g^{δε})_{,α}(a_ε·a_{ij}) + g^{δε}(a_{εα}·a_{ij} + a_ε·a_{ijα}).
        auto Gam_d = [&](std::size_t delta, std::size_t i,
                         std::size_t j, std::size_t alpha) -> T {
            T s = T(0);
            for (std::size_t eps = 0; eps < d; ++eps) {
                s += Gup_d[delta][eps][alpha] * dot1[eps][i][j]
                   + G(delta, eps) * (dot2[eps][alpha][i][j]
                                    + dot3[eps][i][j][alpha]);
            }
            return s;
        };

        // c^δ = g^{ij} Γ^δ_{ij} and (c^δ)_{,α}.
        for (std::size_t delta = 0; delta < d; ++delta) {
            T cv = T(0);
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = 0; j < d; ++j)
                    cv += G(i, j) * Gam(delta, i, j);
            aux.c[delta](q) = cv;

            for (std::size_t alpha = 0; alpha < d; ++alpha) {
                T s = T(0);
                for (std::size_t i = 0; i < d; ++i)
                    for (std::size_t j = 0; j < d; ++j)
                        s += Gup_d[i][j][alpha] * Gam(delta, i, j)
                           + G(i, j) * Gam_d(delta, i, j, alpha);
                aux.c_d[delta][alpha](q) = s;
            }
        }
    }
    return aux;
}

// === Template Instantiations ========================================================

template LaplaceGradAux<double, 2> compute_laplace_grad_aux<double, 2>(
    const IntrinsicGeometry<double, 2>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template LaplaceGradAux<float, 2> compute_laplace_grad_aux<float, 2>(
    const IntrinsicGeometry<float, 2>&);
#endif

} // namespace pyck
