#include "catch.hpp"

#include <memory>

#include "patch.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "gauss_legendre.hpp"
#include "element_values.hpp"
#include "covariant_hessian.hpp"
#include "laplace_beltrami.hpp"
#include "covariant_gradient.hpp"
#include "laplace_beltrami_gradient.hpp"

using namespace pyck;

// ---------------------------------------------------------------------------
// The ElementValues kernels (H, L, D, P) must reproduce the inline reference
// arithmetic the elements use today. The reference forms the connection
// *explicitly from base vectors* (an independent connection oracle) for
// H, an independent metric contraction for D/L, and the 4p's explicit
// lap_1/lap_2 expression for P — so a packing/indexing mismatch in any kernel is
// caught here. Run on a
// degree-3, curved, non-separable patch so the connection and A_{1,2} are
// nonzero (a flat/separable patch would not exercise these terms).
// ---------------------------------------------------------------------------
TEST_CASE("ElementValues kernels match inline reference (curved p=3 patch)",
          "[elements][kernels]")
{
    const Index p = 3, n = 4;                  // single Bézier element, 4 basis/dir
    auto kv  = KnotVector<double>::clamped_uniform(p, n);
    auto bsp = std::make_shared<BSpline<double>>(p, kv);

    // 4×4 control net on a curved, non-separable surface z = ½x² + 0.3xy + 0.4y².
    ColMatrix<double, 3> P(n * n, 3);
    for (Index iv = 0; iv < n; ++iv)
        for (Index iu = 0; iu < n; ++iu) {
            const double x = double(iu) / (n - 1);
            const double y = double(iv) / (n - 1);
            P.row(iu + iv * n) << x, y, 0.5 * x * x + 0.3 * x * y + 0.4 * y * y;
        }
    auto patch = std::make_shared<Patch<double, 2>>(bsp, bsp, P);

    GaussLegendre<double, 2> gauss(p + 1);
    const unsigned flags = Flags::Deriv3;
    ElementValues<double, 2> ev(*patch, 3, flags, gauss);
    ev.reinit(0);

    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();

    const double tol = 1e-9;

    for (Index q = 0; q < Q; ++q)
    {
        auto N1 = ev.results_[1].col(q);   // N_u, N_v
        auto N2 = ev.results_[2].col(q);   // N_uu, N_vv, N_uv
        auto N3 = ev.results_[3].col(q);   // N_uuu, N_uuv, N_uvv, N_vvv

        const operators::CovariantHessian<double, 2>  hess {ev.results_, ev.position_data, ev.g_inv_data, q};
        const operators::LaplaceBeltrami<double, 2>   lapb {ev.results_, ev.position_data, ev.g_inv_data, q};
        const operators::CovariantGradient<double, 2> vgrad{ev.results_, ev.position_data, q};
        const operators::LaplaceBeltramiGradient<double, 2> lgrad{ev.results_, ev.position_data, ev.g_inv_data, q};
        const auto aux = operators::compute_laplace_beltrami_grad_conn<double, 2>(ev.position_data, ev.g_inv_data, q);

        const double gi00 = ev.g_inv(0, 0)(q), gi01 = ev.g_inv(0, 1)(q), gi11 = ev.g_inv(1, 1)(q);

        for (Index i = 0; i < N; ++i)
        {
            const double Ni  = ev.results_[0].col(q)(i);
            const double Nu  = N1(i * 2 + 0),  Nv  = N1(i * 2 + 1);
            const double Nuu = N2(i * 3 + 0),  Nvv = N2(i * 3 + 1),  Nuv = N2(i * 3 + 2);
            const double Nuuu = N3(i * 4 + 0), Nuuv = N3(i * 4 + 1),
                         Nuvv = N3(i * 4 + 2), Nvvv = N3(i * 4 + 3);

            // --- H: covariant Hessian via an explicit base-vector connection ----
            auto H_ref = [&](Index a, Index b) {
                const double Nab = N2(i * 3 + pack2<2>(a, b));
                // Γ_{γ,ab} = A_γ·A_{,ab}; raise to second kind Γ^m_{ab} = g^{mγ}Γ_{γ,ab}.
                const double g1 = ev.a(0).row(q).dot(ev.a_d1(a, b).row(q));
                const double g2 = ev.a(1).row(q).dot(ev.a_d1(a, b).row(q));
                const double Gam0 = gi00 * g1 + gi01 * g2;   // Γ^0_{ab}
                const double Gam1 = gi01 * g1 + gi11 * g2;   // Γ^1_{ab}
                return Nab - Gam0 * Nu - Gam1 * Nv;
            };
            const double H00 = H_ref(0, 0), H11 = H_ref(1, 1), H01 = H_ref(0, 1);
            CHECK(hess(i, 0, 0) == Approx(H00).margin(tol));
            CHECK(hess(i, 1, 1) == Approx(H11).margin(tol));
            CHECK(hess(i, 0, 1) == Approx(H01).margin(tol));

            // --- L: Laplace–Beltrami = g^{αβ} H_{αβ} ----------------------------
            const double L_ref = gi00 * H00 + gi11 * H11 + 2.0 * gi01 * H01;
            CHECK(lapb(i) == Approx(L_ref).margin(tol));

            // --- D: D_{λαβ} = g_{αλ} N_{,β} + (A_α·A_{λ,β}) N --------------------
            for (Index lam = 0; lam < 2; ++lam)
                for (Index a = 0; a < 2; ++a)
                    for (Index b = 0; b < 2; ++b) {
                        const double Nb = N1(i * 2 + b);
                        const double conn = ev.a(a).row(q).dot(ev.a_d1(lam, b).row(q));
                        const double D_ref = ev.g(a, lam)(q) * Nb + conn * Ni;
                        CHECK(vgrad(i, lam, a, b) == Approx(D_ref).margin(tol));
                    }

            // --- P: ∂_α(g^{μν}H_{μν}) via the explicit lap_1/lap_2 formula -------
            const double G11_d1 = aux.G_inv_d[0][0][0], G12_d1 = aux.G_inv_d[0][1][0],
                         G22_d1 = aux.G_inv_d[1][1][0];
            const double G11_d2 = aux.G_inv_d[0][0][1], G12_d2 = aux.G_inv_d[0][1][1],
                         G22_d2 = aux.G_inv_d[1][1][1];
            const double c1 = aux.c[0], c2 = aux.c[1];
            const double c1_d1 = aux.c_d[0][0], c2_d1 = aux.c_d[1][0];
            const double c1_d2 = aux.c_d[0][1], c2_d2 = aux.c_d[1][1];

            const double P0_ref = G11_d1 * Nuu + 2.0 * G12_d1 * Nuv + G22_d1 * Nvv
                                + gi00 * Nuuu + 2.0 * gi01 * Nuuv + gi11 * Nuvv
                                - c1_d1 * Nu - c2_d1 * Nv - c1 * Nuu - c2 * Nuv;
            const double P1_ref = G11_d2 * Nuu + 2.0 * G12_d2 * Nuv + G22_d2 * Nvv
                                + gi00 * Nuuv + 2.0 * gi01 * Nuvv + gi11 * Nvvv
                                - c1_d2 * Nu - c2_d2 * Nv - c1 * Nuv - c2 * Nvv;
            CHECK(lgrad(i, 0) == Approx(P0_ref).margin(tol));
            CHECK(lgrad(i, 1) == Approx(P1_ref).margin(tol));
        }
    }

    // Guard: the patch must actually exercise the connection / off-diagonal
    // curvature — otherwise the test is vacuous (cf. the cylinder blind spot).
    double max_conn = 0.0;
    for (Index q = 0; q < Q; ++q)
        max_conn = std::max(max_conn, std::abs(ev.a(0).row(q).dot(ev.a_d1(1, 1).row(q))));
    CHECK(max_conn > 1e-3);
}
