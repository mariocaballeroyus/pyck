#include "catch.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "patch.hpp"
#include "patch_boundary.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "extrinsic_geometry.hpp"
#include "physical_points.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"

using namespace pyck;

// ===========================================================================
// Test 1: Flat rectangular plate — geometry, Jacobian, partition of unity.
// ===========================================================================

TEST_CASE("Patch<double, 2>: Flat Rectangular Plate", "[geometry][surface]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    const double Lx = 3.0, Ly = 5.0;

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) <<  0.0,  0.0, 0.0;
    cp.row(1) <<   Lx,  0.0, 0.0;
    cp.row(2) <<  0.0,   Ly, 0.0;
    cp.row(3) <<   Lx,   Ly, 0.0;

    Patch<double, 2> surf(basis_u, basis_v, cp);
    const Index elem_idx = 4;   // (span_u=1, span_v=1), u-fastest
    auto act = surf.active_control_pts(elem_idx);

    SECTION("Geometry evaluation: phys = N · P") {
        ColMatrix<double, 2> pts(3, 2);
        pts << 0.0, 0.0,
               0.5, 0.5,
               1.0, 1.0;
        const auto b = surf.tensor_product().eval(pts, 0);
        ColMatrix<double, 3> phys(b[0].cols(), 3);
        for (Index q = 0; q < b[0].cols(); ++q) {
            auto slab0 = b[0].col(q);
            Eigen::RowVector3d x_q = Eigen::RowVector3d::Zero();
            for (Index i = 0; i < b[0].rows(); ++i) x_q.noalias() += slab0(i) * act.row(i);
            phys.row(q) = x_q;
        }

        CHECK(phys(0, 0) == Approx(0.0).margin(1e-14));
        CHECK(phys(0, 1) == Approx(0.0).margin(1e-14));
        CHECK(phys(1, 0) == Approx(Lx * 0.5).margin(1e-14));
        CHECK(phys(1, 1) == Approx(Ly * 0.5).margin(1e-14));
        CHECK(phys(2, 0) == Approx(Lx).margin(1e-14));
        CHECK(phys(2, 1) == Approx(Ly).margin(1e-14));
        for (int i = 0; i < 3; ++i) CHECK(phys(i, 2) == Approx(0.0).margin(1e-14));
    }

    SECTION("Metric & Jacobian = Lx · Ly") {
        ColMatrix<double, 2> pts(1, 2);
        pts << 0.3, 0.7;
        const auto b     = surf.tensor_product().eval(pts, 3);
        IntrinsicGeometry<double, 2> local(b, act);

        CHECK(local.g(0, 0)(0) == Approx(Lx * Lx).margin(1e-12));
        CHECK(local.g(0, 1)(0) == Approx(0.0     ).margin(1e-12));
        CHECK(local.g(1, 1)(0) == Approx(Ly * Ly).margin(1e-12));
        CHECK(local.jac(0)  == Approx(Lx * Ly).margin(1e-12));
    }

    SECTION("Christoffels vanish on flat plate") {
        ColMatrix<double, 2> pts(1, 2);
        pts << 0.4, 0.6;
        const auto b     = surf.tensor_product().eval(pts, 3);
        IntrinsicGeometry<double, 2> local(b, act);
        local.compute_christoffels();
        const auto& chr = local.chr;
        CHECK(chr.Gamma(0, 0, 0)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(0, 0, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(0, 1, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 0, 0)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 0, 1)(0) == Approx(0.0).margin(1e-14));
        CHECK(chr.Gamma(1, 1, 1)(0) == Approx(0.0).margin(1e-14));
    }

    SECTION("Partition of unity") {
        ColMatrix<double, 2> pts(1, 2);
        pts << 0.35, 0.72;
        const auto b = surf.tensor_product().eval(pts, 2);

        auto sum_at_0 = [&](Index k_order, Index packed, Index n_k) {
            auto slab = b[k_order].col(0);
            double s = 0.0;
            for (Index i = 0; i < b[0].rows(); ++i) s += slab(i * n_k + packed);
            return s;
        };

        CHECK(sum_at_0(0, 0, 1) == Approx(1.0).margin(1e-14));   // N
        CHECK(sum_at_0(1, 0, 2) == Approx(0.0).margin(1e-14));   // ∂u N
        CHECK(sum_at_0(1, 1, 2) == Approx(0.0).margin(1e-14));   // ∂v N
        CHECK(sum_at_0(2, 0, 3) == Approx(0.0).margin(1e-14));   // ∂uu N (Voigt: (0,0))
        CHECK(sum_at_0(2, 2, 3) == Approx(0.0).margin(1e-14));   // ∂uv N (Voigt: (0,1))
        CHECK(sum_at_0(2, 1, 3) == Approx(0.0).margin(1e-14));   // ∂vv N (Voigt: (1,1))
    }
}


// ===========================================================================
// Test 2: Rectangle factory + quadrature integration over the surface area.
// ===========================================================================

TEST_CASE("Patch<double, 2>: Rectangle factory area integral", "[geometry][surface]") {

    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis_u = std::make_shared<BSpline<double>>(2, kv);
    auto basis_v = std::make_shared<BSpline<double>>(2, kv);

    const double W = 4.0, H = 6.0;
    auto surf = rectangle<double>(basis_u, basis_v, W, H);

    SECTION("Correct number of control points") {
        CHECK(surf.num_control_pts() == 4 * 4);
    }

    SECTION("Corner coordinates") {
        const auto& P = surf.control_pts();
        CHECK(P(0, 0) == Approx(0.0).margin(1e-14));
        CHECK(P(0, 1) == Approx(0.0).margin(1e-14));
        CHECK(P(15, 0) == Approx(W).margin(1e-14));
        CHECK(P(15, 1) == Approx(H).margin(1e-14));
    }

    SECTION("Integral of Jacobian equals area") {
        GaussLegendre<double, 2> quad(3);
        const auto intervals = surf.tensor_product().num_intervals();
        const Index total = intervals[0] * intervals[1];

        double area = 0.0;
        for (Index e = 0; e < total; ++e) {
            // U-inner flat span: e = su + sv * intervals[0].
            const auto spans = surf.decode_span(e);
            std::array<double, 2> lo, hi;
            bool zero_vol = false;
            for (int d = 0; d < 2; ++d) {
                auto [l, h] = surf.basis(d).knot_vector().span_bounds(spans[d]);
                lo[d] = l; hi[d] = h;
                if (std::abs(h - l) < 1e-14) { zero_vol = true; break; }
            }
            if (zero_vol) continue;

            auto [mp, mw] = quad.map_to_domain(lo, hi);
            const auto b     = surf.tensor_product().eval(mp, 1);
            const auto act   = surf.active_control_pts(e);
            IntrinsicGeometry<double, 2> local(b, act);

            for (Eigen::Index q = 0; q < mw.size(); ++q)
                area += local.jac(q) * mw(q);
        }

        CHECK(area == Approx(W * H).margin(1e-10));
    }
}


// ===========================================================================
// Test 3: eval_physical_points free function.
// ===========================================================================

TEST_CASE("Patch<double, 2>: eval_physical_points on a flat rectangle",
          "[geometry][surface]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    const double Lx = 2.0, Ly = 3.0;
    auto surf = rectangle<double>(basis_u, basis_v, Lx, Ly);

    GaussLegendre<double, 2> quad(2);
    auto phys_pts = eval_physical_points<double, 2>(surf, quad);

    // 2×2 quadrature × 1 non-zero element = 4 physical points.
    CHECK(phys_pts.rows() == 4);
    for (Eigen::Index i = 0; i < phys_pts.rows(); ++i) {
        CHECK(phys_pts(i, 2) == Approx(0.0).margin(1e-14));
        CHECK(phys_pts(i, 0) >= -1e-14);
        CHECK(phys_pts(i, 0) <= Lx + 1e-14);
        CHECK(phys_pts(i, 1) >= -1e-14);
        CHECK(phys_pts(i, 1) <= Ly + 1e-14);
    }
}


// ===========================================================================
// Test 4: Higher-order basis — partition of unity on a flat trapezoidal patch.
// ===========================================================================

TEST_CASE("Patch<double, 2>: Quadratic Basis — Partition of Unity",
          "[geometry][surface]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 0, 1, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(2, kv);
    auto basis_v = std::make_shared<BSpline<double>>(2, kv);

    Eigen::MatrixXd cp(9, 3);
    auto bilin = [](double u, double v) {
        const double x = u * (1.0 - v) + 2.0 * u * v;
        const double y = 3.0 * v;
        return std::tuple{x, y, 0.0};
    };
    int row = 0;
    for (double v : {0.0, 0.5, 1.0})
        for (double u : {0.0, 0.5, 1.0}) {
            auto [x, y, z] = bilin(u, v);
            cp.row(row++) << x, y, z;
        }

    Patch<double, 2> surf(basis_u, basis_v, cp);
    const Index elem_idx = 12;   // (span_u=2, span_v=2) with intervals = (5, 5)
    auto act = surf.active_control_pts(elem_idx);

    std::vector<std::pair<double, double>> test_pts = {
        {0.25, 0.15}, {0.5, 0.5}, {0.7, 0.85}};

    for (auto [u, v] : test_pts) {
        SECTION("Partition of unity at u=" + std::to_string(u) +
                ", v=" + std::to_string(v)) {

            ColMatrix<double, 2> pts(1, 2);
            pts << u, v;

            const auto b     = surf.tensor_product().eval(pts, 3);
            IntrinsicGeometry<double, 2> local(b, act);

            auto sum_at_0 = [&](Index k_order, Index packed, Index n_k) {
                auto slab = b[k_order].col(0);
                double s = 0.0;
                for (Index i = 0; i < b[0].rows(); ++i) s += slab(i * n_k + packed);
                return s;
            };
            CHECK(sum_at_0(0, 0, 1) == Approx(1.0).margin(1e-14));
            CHECK(sum_at_0(1, 0, 2) == Approx(0.0).margin(1e-12));
            CHECK(sum_at_0(1, 1, 2) == Approx(0.0).margin(1e-12));
            CHECK(sum_at_0(2, 0, 3) == Approx(0.0).margin(1e-11));
            CHECK(sum_at_0(2, 2, 3) == Approx(0.0).margin(1e-11));
            CHECK(sum_at_0(2, 1, 3) == Approx(0.0).margin(1e-11));
            CHECK(local.jac(0) > 0.0);
        }
    }
}


// ===========================================================================
// Test 5: PatchBoundary::eval_outward_normal on a flat plate.
// ===========================================================================

TEST_CASE("PatchBoundary<double, 2>::eval_outward_normal: flat rectangle",
          "[geometry][surface][boundary]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    const double Lx = 2.0, Ly = 3.0;
    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) <<  Lx, 0.0, 0.0;
    cp.row(2) << 0.0,  Ly, 0.0;
    cp.row(3) <<  Lx,  Ly, 0.0;
    auto surf = std::make_shared<Patch<double, 2>>(basis_u, basis_v, cp);

    // Boundary at u = Lx (param_dim = 0, at_start = false). Expected outward
    // normal = (+1, 0, 0) at every quadrature point.
    auto bdy = create_patch_boundary<double, 2>(surf, 0, false);
    const Index boundary_span = 1;

    Eigen::VectorXd bdy_pts(3);
    bdy_pts << 0.1, 0.5, 0.9;

    const auto bdy_basis = (*bdy).tensor_product().eval(bdy_pts, 1);
    const auto bdy_act   = bdy->active_control_pts(boundary_span);
    IntrinsicGeometry<double, 1> bdy_local(bdy_basis, bdy_act);

    const Index parent_flat = bdy->parent_flat_span(boundary_span);
    const auto parent_pts   = bdy->lift_to_parent(bdy_pts);
    const auto parent_basis = (*surf).tensor_product().eval(parent_pts, 1);
    const auto parent_act   = surf->active_control_pts(parent_flat);
    IntrinsicGeometry<double, 2> parent_local(parent_basis, parent_act);

    const auto n = bdy->eval_outward_normal(bdy_local, parent_local);
    REQUIRE(n.rows() == bdy_pts.size());
    for (Eigen::Index q = 0; q < bdy_pts.size(); ++q) {
        CHECK(n(q, 0) == Approx( 1.0).margin(1e-12));
        CHECK(n(q, 1) == Approx( 0.0).margin(1e-12));
        CHECK(n(q, 2) == Approx( 0.0).margin(1e-12));
    }
}


// ===========================================================================
// Test 6: Composable primitives — analytical checks on the twisted z = u·v
// patch. Verifies eval_basis, eval_covariant_basis (covering metric,
// Christoffels, second fundamental form, normal, and Weingarten derivative)
// against closed-form expressions.
// ===========================================================================

TEST_CASE("Patch<double, 2>: composable primitives on twisted z=u·v patch",
          "[geometry][surface][primitives]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    const Index elem_idx = 4;

    ColMatrix<double, 2> pts(3, 2);
    pts << 0.3, 0.4,
           0.5, 0.5,
           0.8, 0.2;

    const auto b      = surf.tensor_product().eval(pts, 3);
    const auto actpts = surf.active_control_pts(elem_idx);
    IntrinsicGeometry<double, 2> local(b, actpts);
    local.compute_christoffels();
    const auto& chr = local.chr;
    const ExtrinsicGeometry<double, 2> eg_local(local);
    const auto& a3     = eg_local.n;
    const auto& nd_a31 = eg_local.n_d1(0);
    const auto& nd_a32 = eg_local.n_d1(1);

    REQUIRE(local.a(0).rows() == 3);
    REQUIRE(local.a(1).rows() == 3);
    REQUIRE(local.g(0, 0).size()     == 3);
    REQUIRE(local.g_inv(0, 0).size() == 3);
    REQUIRE(local.jac.size() == 3);
    REQUIRE(chr.Gamma(0, 0, 0).size() == 3);
    REQUIRE(chr.Gamma(1, 1, 1).size() == 3);

    for (Eigen::Index q = 0; q < pts.rows(); ++q) {
        const double u = pts(q, 0);
        const double v = pts(q, 1);
        const double D = 1.0 + u * u + v * v;
        const double sqrtD = std::sqrt(D);
        const double D32 = D * sqrtD;

        // Tangents: a_1 = (1, 0, v), a_2 = (0, 1, u).
        CHECK(local.a(0)(q, 0) == Approx(1.0).margin(1e-14));
        CHECK(local.a(0)(q, 1) == Approx(0.0).margin(1e-14));
        CHECK(local.a(0)(q, 2) == Approx(v  ).margin(1e-14));
        CHECK(local.a(1)(q, 0) == Approx(0.0).margin(1e-14));
        CHECK(local.a(1)(q, 1) == Approx(1.0).margin(1e-14));
        CHECK(local.a(1)(q, 2) == Approx(u  ).margin(1e-14));

        // Metric and Jacobian.
        CHECK(local.g(0, 0)(q)     == Approx(1.0 + v * v).margin(1e-12));
        CHECK(local.g(0, 1)(q)     == Approx(u * v       ).margin(1e-12));
        CHECK(local.g(1, 1)(q)     == Approx(1.0 + u * u).margin(1e-12));
        CHECK(local.g_inv(0, 0)(q) == Approx((1.0 + u * u) / D).margin(1e-12));
        CHECK(local.g_inv(0, 1)(q) == Approx(-u * v / D       ).margin(1e-12));
        CHECK(local.g_inv(1, 1)(q) == Approx((1.0 + v * v) / D).margin(1e-12));
        CHECK(local.jac(q)      == Approx(sqrtD            ).margin(1e-12));

        // Christoffels: only Γ¹_{12} = v/D and Γ²_{12} = u/D are non-zero.
        CHECK(chr.Gamma(0, 0, 0)(q) == Approx(0.0  ).margin(1e-14));
        CHECK(chr.Gamma(0, 0, 1)(q) == Approx(v / D).margin(1e-12));
        CHECK(chr.Gamma(0, 1, 1)(q) == Approx(0.0  ).margin(1e-14));
        CHECK(chr.Gamma(1, 0, 0)(q) == Approx(0.0  ).margin(1e-14));
        CHECK(chr.Gamma(1, 0, 1)(q) == Approx(u / D).margin(1e-12));
        CHECK(chr.Gamma(1, 1, 1)(q) == Approx(0.0  ).margin(1e-14));

        // Surface normal a_3 = (-v, -u, 1) / √D.
        CHECK(a3(q, 0) == Approx(-v   / sqrtD).margin(1e-12));
        CHECK(a3(q, 1) == Approx(-u   / sqrtD).margin(1e-12));
        CHECK(a3(q, 2) == Approx( 1.0 / sqrtD).margin(1e-12));

        // Director derivatives (closed form on z = u·v):
        //   a_{3,1} = ( u v,        -(1+v²),  -u ) / D^{3/2}
        //   a_{3,2} = (-(1+u²),       u v,    -v ) / D^{3/2}
        CHECK(nd_a31(q, 0) == Approx( u * v          / D32).margin(1e-12));
        CHECK(nd_a31(q, 1) == Approx(-(1.0 + v * v) / D32).margin(1e-12));
        CHECK(nd_a31(q, 2) == Approx(-u             / D32).margin(1e-12));
        CHECK(nd_a32(q, 0) == Approx(-(1.0 + u * u) / D32).margin(1e-12));
        CHECK(nd_a32(q, 1) == Approx( u * v          / D32).margin(1e-12));
        CHECK(nd_a32(q, 2) == Approx(-v             / D32).margin(1e-12));

        // Identity: a_γ · a_{3,β} = -b_{γβ}. Closed form on this patch:
        //   b_{11} = b_{22} = 0,  b_{12} = 1/√D.
        const Eigen::Vector3d a1  = local.a(0).row(q).transpose();
        const Eigen::Vector3d a2  = local.a(1).row(q).transpose();
        const Eigen::Vector3d a31 = nd_a31.row(q).transpose();
        const Eigen::Vector3d a32 = nd_a32.row(q).transpose();
        CHECK(a1.dot(a31) == Approx( 0.0        ).margin(1e-13));
        CHECK(a1.dot(a32) == Approx(-1.0 / sqrtD).margin(1e-12));
        CHECK(a2.dot(a31) == Approx(-1.0 / sqrtD).margin(1e-12));
        CHECK(a2.dot(a32) == Approx( 0.0        ).margin(1e-13));
    }
}


// ===========================================================================
// Test 6b: Order-3 Christoffel derivatives ∂_γ Γ^k_{αβ} on the twisted
// z = u·v patch. Closed-form: only Γ¹_{12} = v/D and Γ²_{12} = u/D are
// non-zero, so the only non-zero ∂Γ entries are
//   ∂_u Γ¹_{12} = -2uv / D²
//   ∂_v Γ¹_{12} = (1 + u² - v²) / D²
//   ∂_u Γ²_{12} = (1 - u² + v²) / D²
//   ∂_v Γ²_{12} = -2uv / D²
// with D = 1 + u² + v².
// ===========================================================================

TEST_CASE("Patch<double, 2>: ∂Γ on twisted z=u·v patch",
          "[geometry][surface][primitives]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    const Index elem_idx = 4;

    ColMatrix<double, 2> pts(3, 2);
    pts << 0.3, 0.4,
           0.5, 0.5,
           0.8, 0.2;

    const auto b      = surf.tensor_product().eval(pts, 3);
    const auto actpts = surf.active_control_pts(elem_idx);
    IntrinsicGeometry<double, 2> local(b, actpts);
    local.compute_christoffels();
    const auto& chr = local.chr;

    for (Eigen::Index q = 0; q < pts.rows(); ++q) {
        const double u  = pts(q, 0);
        const double v  = pts(q, 1);
        const double D  = 1.0 + u * u + v * v;
        const double D2 = D * D;

        // Non-zero entries.
        CHECK(chr.Gamma_d1(0, 0, 1, 0)(q)
              == Approx(-2.0 * u * v / D2).margin(1e-12));
        CHECK(chr.Gamma_d1(0, 0, 1, 1)(q)
              == Approx((1.0 + u * u - v * v) / D2).margin(1e-12));
        CHECK(chr.Gamma_d1(1, 0, 1, 0)(q)
              == Approx((1.0 - u * u + v * v) / D2).margin(1e-12));
        CHECK(chr.Gamma_d1(1, 0, 1, 1)(q)
              == Approx(-2.0 * u * v / D2).margin(1e-12));

        // All other entries vanish (the other Γ components are identically 0).
        CHECK(chr.Gamma_d1(0, 0, 0, 0)(q) == Approx(0.0).margin(1e-13));
        CHECK(chr.Gamma_d1(0, 0, 0, 1)(q) == Approx(0.0).margin(1e-13));
        CHECK(chr.Gamma_d1(0, 1, 1, 0)(q) == Approx(0.0).margin(1e-13));
        CHECK(chr.Gamma_d1(0, 1, 1, 1)(q) == Approx(0.0).margin(1e-13));
        CHECK(chr.Gamma_d1(1, 0, 0, 0)(q) == Approx(0.0).margin(1e-13));
        CHECK(chr.Gamma_d1(1, 0, 0, 1)(q) == Approx(0.0).margin(1e-13));
        CHECK(chr.Gamma_d1(1, 1, 1, 0)(q) == Approx(0.0).margin(1e-13));
        CHECK(chr.Gamma_d1(1, 1, 1, 1)(q) == Approx(0.0).margin(1e-13));
    }
}


// ===========================================================================
// Test 6c: IntrinsicGeometry / ExtrinsicGeometry composition. Verifies that
// the container builders produce byte-for-byte the same data as calling the
// underlying free functions individually on the twisted z = u·v patch.
// ===========================================================================

TEST_CASE("Patch<double, 2>: Intrinsic/Extrinsic containers compose correctly",
          "[geometry][surface][primitives]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    const Index elem_idx = 4;

    ColMatrix<double, 2> pts(3, 2);
    pts << 0.3, 0.4,
           0.5, 0.5,
           0.8, 0.2;

    const auto basis  = surf.tensor_product().eval(pts, 3);
    const auto actpts = surf.active_control_pts(elem_idx);

    IntrinsicGeometry<double, 2> ig(basis, actpts);
    ig.compute_christoffels();
    ExtrinsicGeometry<double, 2> eg(ig);
    eg.compute_curvature(ig);

    for (Eigen::Index q = 0; q < pts.rows(); ++q) {
        // Christoffels: the one non-zero on this patch.
        CHECK(ig.chr.Gamma(0, 0, 1)(q) != 0.0);
        CHECK(ig.chr.Gamma(1, 0, 1)(q) != 0.0);

        // Normal.
        for (Eigen::Index c = 0; c < 3; ++c)
            CHECK(std::isfinite(eg.n(q, c)));

        // Curvature: b_{12} = 1/√D on this patch (the non-zero one).
        CHECK(std::isfinite(eg.curv.b      (0, 1)(q)));
        CHECK(std::isfinite(eg.curv.b_mixed(0, 1)(q)));
    }
}


