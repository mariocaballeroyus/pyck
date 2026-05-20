#include "catch.hpp"

#include <cmath>
#include <vector>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "gauss_legendre.hpp"

using namespace pyck;

// ===========================================================================
// Test 1: Flat axis-aligned box — geometry, Jacobian, partition of unity,
//         vanishing Christoffel symbols.
// ===========================================================================

TEST_CASE("Patch<double, 3>: Flat Box", "[geometry][volume]")
{
    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);
    auto basis_w = std::make_shared<BSpline<double>>(1, kv);

    const double Lx = 2.0, Ly = 3.0, Lz = 5.0;

    Eigen::MatrixXd cp(8, 3);
    // Flat layout: i + j*n_u + k*n_u*n_v with n_u = n_v = n_w = 2.
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                cp.row(i + 2 * j + 4 * k) << Lx * i, Ly * j, Lz * k;
            }
        }
    }

    Patch<double, 3> vol(basis_u, basis_v, basis_w, cp);

    // For degree=1, num_basis=2 → 3 spans per direction (incl. zero-width clamped).
    // The single non-degenerate element is at (span_u, span_v, span_w) = (1, 1, 1),
    // flat-encoded u-inner as 1 + 1*3 + 1*3*3 = 13.
    const Index elem_idx = 13;
    auto act = vol.active_control_pts(elem_idx);

    SECTION("Topology / geometric dimensions") {
        CHECK(vol.tdim() == 3);
        CHECK(vol.gdim() == 3);
        CHECK(vol.num_control_pts() == 8);
    }

    SECTION("Metric and Jacobian on a stretched box") {
        ColMatrix<double, 3> pts(1, 3);
        pts << 0.3, 0.5, 0.7;
        const auto b     = vol.tensor_product().eval(pts, 3);
        IntrinsicGeometry<double, 3> local(b, act);

        // Metric layout: (g_11, g_12, g_13, g_22, g_23, g_33).
        CHECK(local.g(0, 0)(0) == Approx(Lx * Lx).margin(1e-12));
        CHECK(local.g(0, 1)(0) == Approx(0.0     ).margin(1e-12));
        CHECK(local.g(0, 2)(0) == Approx(0.0     ).margin(1e-12));
        CHECK(local.g(1, 1)(0) == Approx(Ly * Ly).margin(1e-12));
        CHECK(local.g(1, 2)(0) == Approx(0.0     ).margin(1e-12));
        CHECK(local.g(2, 2)(0) == Approx(Lz * Lz).margin(1e-12));
        CHECK(local.jac(0)  == Approx(Lx * Ly * Lz).margin(1e-12));

        // Inverse metric: diagonal entries are reciprocals of diagonal g.
        CHECK(local.g_inv(0, 0)(0) == Approx(1.0 / (Lx * Lx)).margin(1e-12));
        CHECK(local.g_inv(1, 1)(0) == Approx(1.0 / (Ly * Ly)).margin(1e-12));
        CHECK(local.g_inv(2, 2)(0) == Approx(1.0 / (Lz * Lz)).margin(1e-12));
        CHECK(local.g_inv(0, 1)(0) == Approx(0.0).margin(1e-12));
        CHECK(local.g_inv(0, 2)(0) == Approx(0.0).margin(1e-12));
        CHECK(local.g_inv(1, 2)(0) == Approx(0.0).margin(1e-12));
    }

    SECTION("Christoffels vanish on flat box") {
        ColMatrix<double, 3> pts(1, 3);
        pts << 0.4, 0.6, 0.2;
        const auto b     = vol.tensor_product().eval(pts, 3);
        IntrinsicGeometry<double, 3> local(b, act);
        local.compute_christoffels();
        const auto& chr = local.chr;
        CHECK(chr.Gamma(0, 0, 0)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(0, 0, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(0, 0, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(0, 1, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(0, 1, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(0, 2, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 0, 0)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 0, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 0, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 1, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 1, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 2, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(2, 0, 0)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(2, 0, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(2, 0, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(2, 1, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(2, 1, 2)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(2, 2, 2)(0) == Approx(0.0).margin(1e-14));
    }

    SECTION("Partition of unity (3D)") {
        ColMatrix<double, 3> pts(1, 3);
        pts << 0.35, 0.72, 0.18;
        const auto b = vol.tensor_product().eval(pts, 2);

        auto sum_at_0 = [&](Index k_order, Index packed, Index n_k) {
            auto slab = b[k_order].col(0);
            double s = 0.0;
            for (Index i = 0; i < b[0].rows(); ++i) s += slab(i * n_k + packed);
            return s;
        };
        CHECK(sum_at_0(0, 0, 1) == Approx(1.0).margin(1e-14));   // N
        CHECK(sum_at_0(1, 0, 3) == Approx(0.0).margin(1e-14));   // ∂u
        CHECK(sum_at_0(1, 1, 3) == Approx(0.0).margin(1e-14));   // ∂v
        CHECK(sum_at_0(1, 2, 3) == Approx(0.0).margin(1e-14));   // ∂w
        CHECK(sum_at_0(2, 0, 6) == Approx(0.0).margin(1e-14));   // ∂uu (Voigt: (0,0))
        CHECK(sum_at_0(2, 3, 6) == Approx(0.0).margin(1e-14));   // ∂uv (Voigt: (0,1))
        CHECK(sum_at_0(2, 4, 6) == Approx(0.0).margin(1e-14));   // ∂uw (Voigt: (0,2))
        CHECK(sum_at_0(2, 1, 6) == Approx(0.0).margin(1e-14));   // ∂vv (Voigt: (1,1))
        CHECK(sum_at_0(2, 5, 6) == Approx(0.0).margin(1e-14));   // ∂vw (Voigt: (1,2))
        CHECK(sum_at_0(2, 2, 6) == Approx(0.0).margin(1e-14));   // ∂ww (Voigt: (2,2))
    }
}


// ===========================================================================
// Test 2: Box factory + quadrature integration over the volume.
// ===========================================================================

TEST_CASE("Patch<double, 3>: Box factory volume integral", "[geometry][volume]")
{
    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis_u = std::make_shared<BSpline<double>>(2, kv);
    auto basis_v = std::make_shared<BSpline<double>>(2, kv);
    auto basis_w = std::make_shared<BSpline<double>>(2, kv);

    const double Lx = 4.0, Ly = 6.0, Lz = 2.5;
    auto vol = box<double>(basis_u, basis_v, basis_w, Lx, Ly, Lz);

    SECTION("Number of control points") {
        CHECK(vol.num_control_pts() == 4 * 4 * 4);
    }

    SECTION("Corner coordinates") {
        const auto& P = vol.control_pts();
        CHECK(P(0, 0) == Approx(0.0).margin(1e-14));
        CHECK(P(0, 1) == Approx(0.0).margin(1e-14));
        CHECK(P(0, 2) == Approx(0.0).margin(1e-14));
        // Last cp = (Lx, Ly, Lz).
        const Index last = vol.num_control_pts() - 1;
        CHECK(P(last, 0) == Approx(Lx).margin(1e-14));
        CHECK(P(last, 1) == Approx(Ly).margin(1e-14));
        CHECK(P(last, 2) == Approx(Lz).margin(1e-14));
    }

    SECTION("Integral of Jacobian equals volume") {
        GaussLegendre<double, 3> quad(3);
        const auto intervals = vol.tensor_product().num_intervals();
        const Index total = intervals[0] * intervals[1] * intervals[2];

        double vol_int = 0.0;
        for (Index e = 0; e < total; ++e)
        {
            const auto spans = vol.decode_span(e);
            std::array<double, 3> lo{}, hi{};
            bool zero_vol = false;
            for (std::size_t d = 0; d < 3; ++d) {
                auto [l, h] = vol.basis(d).knot_vector().span_bounds(spans[d]);
                lo[d] = l; hi[d] = h;
                if (std::abs(h - l) < 1e-14) { zero_vol = true; break; }
            }
            if (zero_vol) continue;

            auto [mp, mw] = quad.map_to_domain(lo, hi);
            const auto b     = vol.tensor_product().eval(mp, 1);
            const auto act   = vol.active_control_pts(e);
            IntrinsicGeometry<double, 3> local(b, act);

            for (Eigen::Index q = 0; q < mw.size(); ++q)
                vol_int += local.jac(q) * mw(q);
        }

        CHECK(vol_int == Approx(Lx * Ly * Lz).margin(1e-10));
    }
}


// ===========================================================================
// Test 3: decode_span / active_control_pts on a non-symmetric grid.
// ===========================================================================

TEST_CASE("Patch<double, 3>: decode_span and active_control_pts",
          "[geometry][volume]")
{
    auto kv_u = KnotVector<double>::clamped_uniform(2, 4);   // 6 spans (4 zero-width, 2 non-deg)
    auto kv_v = KnotVector<double>::clamped_uniform(2, 5);   // 7 spans (4 zero-width, 3 non-deg)
    auto kv_w = KnotVector<double>::clamped_uniform(2, 4);   // 6 spans (4 zero-width, 2 non-deg)

    auto bu = std::make_shared<BSpline<double>>(2, kv_u);
    auto bv = std::make_shared<BSpline<double>>(2, kv_v);
    auto bw = std::make_shared<BSpline<double>>(2, kv_w);

    auto vol = box<double>(bu, bv, bw, 1.0, 1.0, 1.0);

    // num_intervals() reports all knot spans, including zero-width clamped ones.
    const auto intervals = vol.tensor_product().num_intervals();
    REQUIRE(intervals[0] == 6);
    REQUIRE(intervals[1] == 7);
    REQUIRE(intervals[2] == 6);

    SECTION("decode_span round-trips lexicographic encoding") {
        const Index total = intervals[0] * intervals[1] * intervals[2];
        for (Index e = 0; e < total; ++e)
        {
            auto spans = vol.decode_span(e);
            // Re-encode and check.
            const Index re = spans[0]
                           + spans[1] * intervals[0]
                           + spans[2] * intervals[0] * intervals[1];
            CHECK(re == e);
        }
    }

    SECTION("active_control_pts returns (p+1)^3 = 27 points per non-degenerate element") {
        for (Index e = 0; e < intervals[0] * intervals[1] * intervals[2]; ++e)
        {
            auto spans = vol.decode_span(e);
            bool zero_width = false;
            for (std::size_t k = 0; k < 3; ++k) {
                auto [lo, hi] = vol.basis(k).knot_vector().span_bounds(spans[k]);
                if (std::abs(hi - lo) < 1e-14) { zero_width = true; break; }
            }
            if (zero_width) continue;

            const auto act = vol.active_control_pts(e);
            CHECK(act.rows() == 27);
            CHECK(act.cols() == 3);
        }
    }
}
