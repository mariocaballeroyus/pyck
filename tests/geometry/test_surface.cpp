#include "catch.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "patch.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knots.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"

using namespace pyck;

// ===========================================================================
// Test 1: Flat surface — no Christoffel corrections
// ===========================================================================

TEST_CASE("Patch<double, 2>: Flat Rectangular Plate", "[geometry][surface]") {

    // p=1 bilinear basis in both directions
    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    double Lx = 3.0, Ly = 5.0;

    // Flat rectangle: x = Lx*u, y = Ly*v, z = 0
    // Control points in u-fastest order (i_u + i_v * n_u):
    //   (0,0) -> (0, 0, 0)    (1,0) -> (Lx, 0, 0)
    //   (0,1) -> (0, Ly, 0)   (1,1) -> (Lx, Ly, 0)
    Eigen::MatrixXd cp(4, 3);
    cp.row(0) <<  0.0,  0.0, 0.0;   // (iu=0, iv=0)
    cp.row(1) <<   Lx,  0.0, 0.0;   // (iu=1, iv=0)
    cp.row(2) <<  0.0,   Ly, 0.0;   // (iu=0, iv=1)
    cp.row(3) <<   Lx,   Ly, 0.0;   // (iu=1, iv=1)

    Patch<double, 2> surf(basis_u, basis_v, cp);

    // Single non-zero element for clamped-uniform p=1, n=2
    // Knot spans: u has 3 spans (indices 0,1,2 → only 1 is non-zero)
    //             v has 3 spans (indices 0,1,2 → only 1 is non-zero)
    // The non-zero element is at span_u=1, span_v=1 → flat index = 1*3 + 1 = 4
    // Actually, for p=1 with knots {0,0,1,1}: num_spans = 3, non-zero is span 1
    // intervals = {3, 3}, flat elem_idx = span_u * intervals_v + span_v = 1*3 + 1 = 4
    Index elem_idx = 4;

    SECTION("Geometry Evaluation") {
        Eigen::MatrixXd pts(3, 2);
        pts << 0.0, 0.0,
               0.5, 0.5,
               1.0, 1.0;

        auto geom = surf.eval_geometry(pts, elem_idx);

        CHECK(geom(0, 0) == Approx(0.0).margin(1e-14));
        CHECK(geom(0, 1) == Approx(0.0).margin(1e-14));
        CHECK(geom(1, 0) == Approx(Lx * 0.5).margin(1e-14));
        CHECK(geom(1, 1) == Approx(Ly * 0.5).margin(1e-14));
        CHECK(geom(2, 0) == Approx(Lx).margin(1e-14));
        CHECK(geom(2, 1) == Approx(Ly).margin(1e-14));

        // z should be zero everywhere
        for (int i = 0; i < 3; ++i)
            CHECK(geom(i, 2) == Approx(0.0).margin(1e-14));
    }

    SECTION("Jacobian = Lx * Ly") {
        Eigen::MatrixXd pts(1, 2);
        pts << 0.3, 0.7;

        auto [result, jac] = surf.eval_shape_functions(pts, elem_idx, 2);

        // Jacobian = sqrt(det(g)) = Lx * Ly for a flat uniform rectangle
        CHECK(jac(0) == Approx(Lx * Ly).margin(1e-12));
    }

    SECTION("No Christoffel corrections — N_{;ab} == N_{,ab}") {
        Eigen::MatrixXd pts(1, 2);
        pts << 0.4, 0.6;

        auto [result, jac] = surf.eval_shape_functions(pts, elem_idx, 2);
        auto basis_derivs = surf.eval_basis_functions(pts, elem_idx, 2);

        // Stride for flat index
        const int S = 3;  // order+1

        // result[3] == N_{,uu}  (index (2,0) in tensor product = 2*S + 0 = 6)
        for (int a = 0; a < 4; ++a) {
            CHECK(result[3](0, a) == Approx(basis_derivs[6](0, a)).margin(1e-14));
        }

        // result[4] == N_{,uv}  (index (1,1) in tensor product = 1*S + 1 = 4)
        for (int a = 0; a < 4; ++a) {
            CHECK(result[4](0, a) == Approx(basis_derivs[4](0, a)).margin(1e-14));
        }

        // result[5] == N_{,vv}  (index (0,2) in tensor product = 0*S + 2 = 2)
        for (int a = 0; a < 4; ++a) {
            CHECK(result[5](0, a) == Approx(basis_derivs[2](0, a)).margin(1e-14));
        }
    }

    SECTION("Partition of unity") {
        Eigen::MatrixXd pts(1, 2);
        pts << 0.35, 0.72;

        auto [result, jac] = surf.eval_shape_functions(pts, elem_idx, 2);

        // Sum of N = 1
        CHECK(result[0].row(0).sum() == Approx(1.0).margin(1e-14));
        // Sum of first derivatives = 0
        CHECK(result[1].row(0).sum() == Approx(0.0).margin(1e-14));
        CHECK(result[2].row(0).sum() == Approx(0.0).margin(1e-14));
        // Sum of second covariant derivatives = 0
        CHECK(result[3].row(0).sum() == Approx(0.0).margin(1e-14));
        CHECK(result[4].row(0).sum() == Approx(0.0).margin(1e-14));
        CHECK(result[5].row(0).sum() == Approx(0.0).margin(1e-14));
    }
}


// ===========================================================================
// Test 2: Twisted surface — non-trivial Christoffel symbols
// ===========================================================================

TEST_CASE("Patch<double, 2>: Twisted Bilinear Plate (z = u*v)", "[geometry][surface]") {

    // p=1 bilinear: x(u,v) = (u, v, u*v)
    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;   // (0,0)
    cp.row(1) << 1.0, 0.0, 0.0;   // (1,0)
    cp.row(2) << 0.0, 1.0, 0.0;   // (0,1)
    cp.row(3) << 1.0, 1.0, 1.0;   // (1,1):  z = 1*1 = 1

    Patch<double, 2> surf(basis_u, basis_v, cp);

    // Non-zero element at span_u=1, span_v=1 → flat index 4
    Index elem_idx = 4;

    std::vector<std::pair<double, double>> test_pts = {
        {0.3, 0.4}, {0.1, 0.9}, {0.5, 0.5}, {0.8, 0.2}};

    for (auto [u, v] : test_pts)
    {
        SECTION("Evaluation at u=" + std::to_string(u) +
                ", v=" + std::to_string(v)) {

            Eigen::MatrixXd pts(1, 2);
            pts << u, v;

            auto [result, jac] = surf.eval_shape_functions(pts, elem_idx, 2);

            // --- Analytical quantities ---
            // Tangent vectors: a_1 = (1,0,v), a_2 = (0,1,u)
            double g11 = 1.0 + v * v;
            double g12 = u * v;
            double g22 = 1.0 + u * u;
            double det_g = g11 * g22 - g12 * g12;  // = 1 + u² + v²
            double J_ana = std::sqrt(det_g);

            CHECK(jac(0) == Approx(J_ana).margin(1e-12));

            // --- Partition of unity ---
            CHECK(result[0].row(0).sum() == Approx(1.0).margin(1e-14));
            CHECK(result[1].row(0).sum() == Approx(0.0).margin(1e-14));
            CHECK(result[2].row(0).sum() == Approx(0.0).margin(1e-14));
            CHECK(result[3].row(0).sum() == Approx(0.0).margin(1e-12));
            CHECK(result[4].row(0).sum() == Approx(0.0).margin(1e-12));
            CHECK(result[5].row(0).sum() == Approx(0.0).margin(1e-12));

            // --- Shape function values (bilinear) ---
            // Tensor product ordering: (u-outer, v-inner)
            //   0=(1-u)(1-v)  1=(1-u)v  2=u(1-v)  3=uv
            double N[4] = {(1.0-u)*(1.0-v), (1.0-u)*v, u*(1.0-v), u*v};
            for (int a = 0; a < 4; ++a)
                CHECK(result[0](0, a) == Approx(N[a]).margin(1e-14));

            // --- First parametric derivatives ---
            double Nu[4] = {-(1.0-v), -v, (1.0-v), v};
            double Nv[4] = {-(1.0-u), (1.0-u), -u, u};
            for (int a = 0; a < 4; ++a) {
                CHECK(result[1](0, a) == Approx(Nu[a]).margin(1e-14));
                CHECK(result[2](0, a) == Approx(Nv[a]).margin(1e-14));
            }

            // --- Analytical Christoffel symbols ---
            // Only Γ^1_{12} and Γ^2_{12} are non-zero for z = u*v
            double inv_det = 1.0 / det_g;
            double gi11 =  g22 * inv_det;
            double gi12 = -g12 * inv_det;
            double gi22 =  g11 * inv_det;

            // a_{12} · a_1 = v,  a_{12} · a_2 = u  (a_{12} = (0,0,1))
            double G1_12 = gi11 * v + gi12 * u;
            double G2_12 = gi12 * v + gi22 * u;

            // --- N_{;uu} and N_{;vv}: Γ_{11} and Γ_{22} vanish ---
            double Nuu[4] = {0.0, 0.0, 0.0, 0.0};
            double Nvv[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                CHECK(result[3](0, a) == Approx(Nuu[a]).margin(1e-14));
                CHECK(result[5](0, a) == Approx(Nvv[a]).margin(1e-14));
            }

            // --- N_{;uv} = N_{,uv} − Γ^1_{12} N_{,u} − Γ^2_{12} N_{,v} ---
            double Nuv_param[4] = {1.0, -1.0, -1.0, 1.0};
            for (int a = 0; a < 4; ++a) {
                double expected = Nuv_param[a] - G1_12 * Nu[a] - G2_12 * Nv[a];
                CHECK(result[4](0, a) == Approx(expected).margin(1e-12));
            }
        }
    }
}


// ===========================================================================
// Test 3: Rectangle factory + quadrature integration
// ===========================================================================

TEST_CASE("Patch<double, 2>: Rectangle Factory", "[geometry][surface]") {

    auto kv = clamped_uniform_knots<double>(2, 4);
    auto basis_u = std::make_shared<BSpline<double>>(2, kv);
    auto basis_v = std::make_shared<BSpline<double>>(2, kv);

    double W = 4.0, H = 6.0;
    auto surf = rectangle<double>(basis_u, basis_v, W, H);

    SECTION("Correct number of control points") {
        CHECK(surf.num_control_pts() == 4 * 4);
    }

    SECTION("Corner coordinates") {
        const auto& P = surf.control_pts();
        // First corner (iu=0, iv=0) should be (0,0,0)
        CHECK(P(0, 0) == Approx(0.0).margin(1e-14));
        CHECK(P(0, 1) == Approx(0.0).margin(1e-14));

        // Last corner (iu=3, iv=3) → global 3 + 3*4 = 15
        CHECK(P(15, 0) == Approx(W).margin(1e-14));
        CHECK(P(15, 1) == Approx(H).margin(1e-14));
    }

    SECTION("Integral of Jacobian equals area") {
        GaussLegendre<double, 2> quad(3);
        auto intervals = surf.tensor_product().num_intervals();
        Index total = intervals[0] * intervals[1];

        double area = 0.0;
        for (Index e = 0; e < total; ++e) {
            std::array<double, 2> lo, hi;
            bool zero_vol = false;
            for (int d = 0; d < 2; ++d) {
                auto [l, h] = surf.basis(d).knot_vector().span_bounds(
                    d == 0 ? e / intervals[1] : e % intervals[1]);
                lo[d] = l; hi[d] = h;
                if (std::abs(h - l) < 1e-14) { zero_vol = true; break; }
            }
            if (zero_vol) continue;

            auto [mp, mw] = quad.map_to_domain(lo, hi);
            auto [res, jac] = surf.eval_shape_functions(mp, e, 1);

            for (int q = 0; q < mw.size(); ++q)
                area += jac(q) * mw(q);
        }

        CHECK(area == Approx(W * H).margin(1e-10));
    }
}


// ===========================================================================
// Test 4: eval_physical_points
// ===========================================================================

TEST_CASE("Patch<double, 2>: Physical Points Evaluation", "[geometry][surface]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    double Lx = 2.0, Ly = 3.0;
    auto surf = rectangle<double>(basis_u, basis_v, Lx, Ly);

    GaussLegendre<double, 2> quad(2);
    auto phys_pts = surf.eval_physical_points(quad);

    // Should have Q = 2*2 = 4 physical points (one non-zero element)
    CHECK(phys_pts.rows() == 4);

    // All z-coords should be zero
    for (int i = 0; i < phys_pts.rows(); ++i)
        CHECK(phys_pts(i, 2) == Approx(0.0).margin(1e-14));

    // x should be in [0, Lx], y in [0, Ly]
    for (int i = 0; i < phys_pts.rows(); ++i) {
        CHECK(phys_pts(i, 0) >= -1e-14);
        CHECK(phys_pts(i, 0) <= Lx + 1e-14);
        CHECK(phys_pts(i, 1) >= -1e-14);
        CHECK(phys_pts(i, 1) <= Ly + 1e-14);
    }
}


// ===========================================================================
// Test 5: Higher-order basis with curved surface
// ===========================================================================

TEST_CASE("Patch<double, 2>: Quadratic Basis — Partition of Unity", "[geometry][surface]") {

    // p=2 quadratic basis in both directions, single element
    auto kv = KnotVector<double>({0, 0, 0, 1, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(2, kv);
    auto basis_v = std::make_shared<BSpline<double>>(2, kv);

    // 3x3 = 9 control points defining a half-cylinder-like surface
    //   x(u,v) = u, y(u,v) = R*sin(pi*v), z(u,v) = R*(1-cos(pi*v))
    // Approximate with quadratic control points
    Eigen::MatrixXd cp(9, 3);
    double R = 2.0;
    // u-fastest: global = iu + iv * 3
    // iv=0 (v=0): P on bottom
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 0.5, 0.0, 0.0;
    cp.row(2) << 1.0, 0.0, 0.0;
    // iv=1 (v=0.5): P at mid-height
    cp.row(3) << 0.0, R, R;
    cp.row(4) << 0.5, R, R;
    cp.row(5) << 1.0, R, R;
    // iv=2 (v=1): P on top
    cp.row(6) << 0.0, 0.0, 2.0*R;
    cp.row(7) << 0.5, 0.0, 2.0*R;
    cp.row(8) << 1.0, 0.0, 2.0*R;

    Patch<double, 2> surf(basis_u, basis_v, cp);

    // Non-zero element: for p=2 with knots {0,0,0,1,1,1}
    // num_spans = 5, and the non-zero span is at index 2 in each direction
    // flat = span_u * 5 + span_v = 2*5 + 2 = 12
    Index elem_idx = 12;

    std::vector<std::pair<double, double>> test_pts = {
        {0.25, 0.15}, {0.5, 0.5}, {0.7, 0.85}};

    for (auto [u, v] : test_pts) {
        SECTION("Partition of unity at u=" + std::to_string(u) +
                ", v=" + std::to_string(v)) {

            Eigen::MatrixXd pts(1, 2);
            pts << u, v;

            auto [result, jac] = surf.eval_shape_functions(pts, elem_idx, 2);

            // Shape functions sum to 1
            CHECK(result[0].row(0).sum() == Approx(1.0).margin(1e-14));

            // All derivatives sum to 0
            CHECK(result[1].row(0).sum() == Approx(0.0).margin(1e-12));
            CHECK(result[2].row(0).sum() == Approx(0.0).margin(1e-12));
            CHECK(result[3].row(0).sum() == Approx(0.0).margin(1e-11));
            CHECK(result[4].row(0).sum() == Approx(0.0).margin(1e-11));
            CHECK(result[5].row(0).sum() == Approx(0.0).margin(1e-11));

            // Jacobian must be positive
            CHECK(jac(0) > 0.0);
        }
    }
}
