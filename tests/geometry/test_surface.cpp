#include "catch.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "patch.hpp"
#include "patch_boundary.hpp"
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

    SECTION("No Christoffel corrections — N_{;ab} == N_{,ab} / sqrt(g_aa * g_bb)") {
        Eigen::MatrixXd pts(1, 2);
        pts << 0.4, 0.6;

        auto [result, jac] = surf.eval_shape_functions(pts, elem_idx, 2);
        auto basis_derivs = surf.eval_basis_functions(pts, elem_idx, 2);

        // Stride for flat index
        const int S = 3;  // order+1

        // For a flat rectangle: g11 = Lx^2, g22 = Ly^2.
        // eval_shape_functions returns metric-normalised covariant derivatives:
        //   result[3] = N_{;uu} / g11        = N_{,uu} / Lx^2  (Christoffels vanish)
        //   result[4] = N_{;uv} / sqrt(g11*g22) = N_{,uv} / (Lx*Ly)
        //   result[5] = N_{;vv} / g22        = N_{,vv} / Ly^2
        const double g11 = Lx * Lx;
        const double g22 = Ly * Ly;
        const double sg  = Lx * Ly;  // sqrt(g11 * g22)

        for (int a = 0; a < 4; ++a) {
            CHECK(result[3](0, a) == Approx(basis_derivs[2 * S + 0](0, a) / g11).margin(1e-14));
            CHECK(result[4](0, a) == Approx(basis_derivs[1 * S + 1](0, a) / sg ).margin(1e-14));
            CHECK(result[5](0, a) == Approx(basis_derivs[0 * S + 2](0, a) / g22).margin(1e-14));
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

            // First-derivative semantics now: planar (x, y) Cartesian via 2D
            // inverse Jacobian. For this curved surface (z = u*v) the planar
            // Jacobian is the identity (∂x/∂u = ∂y/∂v = 1, off-diagonals 0),
            // so result[1] = N_{,u} and result[2] = N_{,v}.
            double Nu[4] = {-(1.0-v), -v, (1.0-v), v};
            double Nv[4] = {-(1.0-u), (1.0-u), -u, u};
            for (int a = 0; a < 4; ++a) {
                CHECK(result[1](0, a) == Approx(Nu[a]).margin(1e-14));
                CHECK(result[2](0, a) == Approx(Nv[a]).margin(1e-14));
            }

            // Second derivatives in (x, y): for the bilinear basis the
            // parametric N_{,uu} and N_{,vv} are zero, and N_{,uv} = ±1.
            // With the planar identity Jacobian and zero second-geometry
            // (x, y) derivatives, result[3]=result[5]=0 and result[4]=N_{,uv}.
            double Nuv[4] = {1.0, -1.0, -1.0, 1.0};
            for (int a = 0; a < 4; ++a) {
                CHECK(result[3](0, a) == Approx(0.0).margin(1e-14));
                CHECK(result[5](0, a) == Approx(0.0).margin(1e-14));
                CHECK(result[4](0, a) == Approx(Nuv[a]).margin(1e-12));
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
// Test 5: Higher-order basis on a flat trapezoidal patch
// ===========================================================================

TEST_CASE("Patch<double, 2>: Quadratic Basis — Partition of Unity", "[geometry][surface]") {

    // p=2 quadratic basis in both directions, single element.
    auto kv = KnotVector<double>({0, 0, 0, 1, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(2, kv);
    auto basis_v = std::make_shared<BSpline<double>>(2, kv);

    // 3x3 = 9 control points defining a flat trapezoidal patch in the (x, y)
    // plane (non-axis-aligned, non-orthogonal parametric coordinates).
    // Bilinear corners (0,0), (1,0), (2,3), (0,3); quadratic by interpolating
    // at Greville (parametric mid) points.
    Eigen::MatrixXd cp(9, 3);
    auto bilin = [](double u, double v) {
        // (1-u)(1-v)·(0,0) + u(1-v)·(1,0) + (1-u)v·(0,3) + uv·(2,3)
        double x = u * (1.0 - v) + 2.0 * u * v;
        double y = 3.0 * v;
        return std::tuple{x, y, 0.0};
    };
    int row = 0;
    for (double v : {0.0, 0.5, 1.0}) {
        for (double u : {0.0, 0.5, 1.0}) {
            auto [x, y, z] = bilin(u, v);
            cp.row(row++) << x, y, z;
        }
    }

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


// ===========================================================================
// Test 6: Patch<T,2>::eval_local_frame on a flat rectangle
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_local_frame: flat rectangle", "[geometry][surface][frame]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    double Lx = 3.0, Ly = 5.0;
    Eigen::MatrixXd cp(4, 3);
    cp.row(0) <<  0.0,  0.0, 0.0;
    cp.row(1) <<   Lx,  0.0, 0.0;
    cp.row(2) <<  0.0,   Ly, 0.0;
    cp.row(3) <<   Lx,   Ly, 0.0;

    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;  // (span_u=1, span_v=1) flat-indexed u-fastest

    Eigen::MatrixXd pts(1, 2);
    pts << 0.3, 0.7;

    auto [a1, a2, a3, jac] = surf.eval_local_frame(pts, elem_idx);

    // Raw covariant tangents have magnitudes = (Lx, Ly)
    CHECK(a1(0, 0) == Approx(Lx).margin(1e-14));
    CHECK(a1(0, 1) == Approx(0.0).margin(1e-14));
    CHECK(a1(0, 2) == Approx(0.0).margin(1e-14));
    CHECK(a2(0, 0) == Approx(0.0).margin(1e-14));
    CHECK(a2(0, 1) == Approx(Ly).margin(1e-14));
    CHECK(a2(0, 2) == Approx(0.0).margin(1e-14));

    // Director is +ẑ
    CHECK(a3(0, 0) == Approx(0.0).margin(1e-14));
    CHECK(a3(0, 1) == Approx(0.0).margin(1e-14));
    CHECK(a3(0, 2) == Approx(1.0).margin(1e-14));

    // Area element = ||a1 × a2|| = Lx * Ly
    CHECK(jac(0) == Approx(Lx * Ly).margin(1e-12));
}


// ===========================================================================
// Test 7: Patch<T,2>::eval_local_frame on a curved (twisted) bilinear patch
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_local_frame: twisted z=u*v patch", "[geometry][surface][frame]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    std::vector<std::pair<double, double>> test_pts = {
        {0.3, 0.4}, {0.5, 0.5}, {0.8, 0.2}};

    for (auto [u, v] : test_pts) {
        SECTION("at u=" + std::to_string(u) + ", v=" + std::to_string(v)) {
            Eigen::MatrixXd pts(1, 2);
            pts << u, v;

            auto [a1, a2, a3, jac] = surf.eval_local_frame(pts, elem_idx);

            // Analytical: a_1 = (1,0,v), a_2 = (0,1,u),
            //             a_3 = (-v,-u,1) / sqrt(1+u^2+v^2)
            CHECK(a1(0, 0) == Approx(1.0).margin(1e-14));
            CHECK(a1(0, 1) == Approx(0.0).margin(1e-14));
            CHECK(a1(0, 2) == Approx(v).margin(1e-14));
            CHECK(a2(0, 0) == Approx(0.0).margin(1e-14));
            CHECK(a2(0, 1) == Approx(1.0).margin(1e-14));
            CHECK(a2(0, 2) == Approx(u).margin(1e-14));

            const double n = std::sqrt(1.0 + u * u + v * v);
            CHECK(a3(0, 0) == Approx(-v / n).margin(1e-12));
            CHECK(a3(0, 1) == Approx(-u / n).margin(1e-12));
            CHECK(a3(0, 2) == Approx( 1.0 / n).margin(1e-12));

            CHECK(jac(0) == Approx(n).margin(1e-12));
        }
    }
}


// ===========================================================================
// Test 8: PatchBoundary<T,2>::eval_local_frame delegates a3 to parent
// ===========================================================================

TEST_CASE("PatchBoundary<double, 2>::eval_local_frame: a3 delegated from parent",
          "[geometry][surface][boundary][frame]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    // Twisted patch z = u*v
    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    auto surf = std::make_shared<Patch<double, 2>>(basis_u, basis_v, cp);

    // Boundary at u = 1 (param_dim = 0, at_start = false)
    auto bdy = create_patch_boundary<double, 2>(surf, 0, false);

    // Single non-zero boundary span = 1
    Index boundary_span = 1;
    Eigen::VectorXd bdy_pts(2);
    bdy_pts << 0.25, 0.75;

    auto [b_a1, b_a2, b_a3, b_jac] = bdy->eval_local_frame(bdy_pts, boundary_span);

    // Parent frame at the lifted points
    auto parent_pts_lifted = bdy->lift_to_parent(bdy_pts);
    Index parent_span = bdy->parent_flat_span(boundary_span);
    auto [p_a1, p_a2, p_a3, p_jac] = surf->eval_local_frame(parent_pts_lifted, parent_span);

    // Boundary a3 must equal parent a3 exactly (the delegation invariant).
    for (Eigen::Index q = 0; q < bdy_pts.size(); ++q) {
        CHECK(b_a3(q, 0) == Approx(p_a3(q, 0)).margin(1e-14));
        CHECK(b_a3(q, 1) == Approx(p_a3(q, 1)).margin(1e-14));
        CHECK(b_a3(q, 2) == Approx(p_a3(q, 2)).margin(1e-14));
    }

    // Boundary tangent at u=1 is ∂x/∂v = (0, 1, u=1), magnitude √2.
    for (Eigen::Index q = 0; q < bdy_pts.size(); ++q) {
        CHECK(b_a1(q, 0) == Approx(0.0).margin(1e-14));
        CHECK(b_a1(q, 1) == Approx(1.0).margin(1e-14));
        CHECK(b_a1(q, 2) == Approx(1.0).margin(1e-14));
        CHECK(b_jac(q) == Approx(std::sqrt(2.0)).margin(1e-14));
    }

    // Orthogonality: a2 ⊥ a1 and a2 ⊥ a3 at every quadrature point.
    for (Eigen::Index q = 0; q < bdy_pts.size(); ++q) {
        const double dot12 = b_a1.row(q).dot(b_a2.row(q));
        const double dot32 = b_a3.row(q).dot(b_a2.row(q));
        CHECK(dot12 == Approx(0.0).margin(1e-12));
        CHECK(dot32 == Approx(0.0).margin(1e-12));
        // a2 is unit
        CHECK(b_a2.row(q).norm() == Approx(1.0).margin(1e-12));
    }
}


// ===========================================================================
// Test 9: PatchBoundary::eval_local_frame.a2 matches legacy eval_outward_normal
// on a flat plate (regression).
// ===========================================================================

TEST_CASE("PatchBoundary<double, 2>: flat-plate regression vs eval_outward_normal",
          "[geometry][surface][boundary][frame]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    double Lx = 2.0, Ly = 3.0;
    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) <<  Lx, 0.0, 0.0;
    cp.row(2) << 0.0,  Ly, 0.0;
    cp.row(3) <<  Lx,  Ly, 0.0;
    auto surf = std::make_shared<Patch<double, 2>>(basis_u, basis_v, cp);

    auto bdy = create_patch_boundary<double, 2>(surf, 0, false);  // u = Lx edge
    Index boundary_span = 1;

    Eigen::VectorXd bdy_pts(3);
    bdy_pts << 0.1, 0.5, 0.9;

    // New frame method
    auto [b_a1, b_a2, b_a3, b_jac] = bdy->eval_local_frame(bdy_pts, boundary_span);

    // Legacy eval_outward_normal needs derivative matrices at the same points.
    auto bdy_derivs = bdy->eval_basis_functions(bdy_pts, boundary_span, 1);
    auto parent_pts_lifted = bdy->lift_to_parent(bdy_pts);
    Index parent_span = bdy->parent_flat_span(boundary_span);
    auto [parent_res, parent_jac] = surf->eval_shape_functions(parent_pts_lifted, parent_span, 1);

    auto n_legacy = bdy->eval_outward_normal(boundary_span, bdy_derivs,
                                             parent_span, parent_res);

    // For u=Lx edge of a flat rectangle, outward normal = (+1, 0, 0).
    for (Eigen::Index q = 0; q < bdy_pts.size(); ++q) {
        CHECK(n_legacy(q, 0) == Approx(1.0).margin(1e-12));
        CHECK(n_legacy(q, 1) == Approx(0.0).margin(1e-12));
        CHECK(n_legacy(q, 2) == Approx(0.0).margin(1e-12));
        CHECK(b_a2(q, 0) == Approx(n_legacy(q, 0)).margin(1e-12));
        CHECK(b_a2(q, 1) == Approx(n_legacy(q, 1)).margin(1e-12));
        CHECK(b_a2(q, 2) == Approx(n_legacy(q, 2)).margin(1e-12));
    }
}


// ===========================================================================
// Test 10: Patch<T,2>::eval_surface_geometry on a flat rectangle
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_surface_geometry: flat rectangle",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    double Lx = 3.0, Ly = 5.0;
    Eigen::MatrixXd cp(4, 3);
    cp.row(0) <<  0.0,  0.0, 0.0;
    cp.row(1) <<   Lx,  0.0, 0.0;
    cp.row(2) <<  0.0,   Ly, 0.0;
    cp.row(3) <<   Lx,   Ly, 0.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    Eigen::MatrixXd pts(1, 2);
    pts << 0.3, 0.7;
    auto [a1, a2, a3, g_inv, b, jac] = surf.eval_surface_geometry(pts, elem_idx);

    // Tangents and director
    CHECK(a1(0, 0) == Approx(Lx).margin(1e-14));
    CHECK(a1(0, 1) == Approx(0.0).margin(1e-14));
    CHECK(a2(0, 0) == Approx(0.0).margin(1e-14));
    CHECK(a2(0, 1) == Approx(Ly).margin(1e-14));
    CHECK(a3(0, 2) == Approx(1.0).margin(1e-14));
    CHECK(jac(0) == Approx(Lx * Ly).margin(1e-12));

    // g_inv = diag(1/Lx², 1/Ly²)
    CHECK(g_inv(0, 0) == Approx(1.0 / (Lx * Lx)).margin(1e-12));
    CHECK(g_inv(0, 1) == Approx(0.0).margin(1e-14));
    CHECK(g_inv(0, 2) == Approx(1.0 / (Ly * Ly)).margin(1e-12));

    // Flat → b vanishes
    CHECK(b(0, 0) == Approx(0.0).margin(1e-14));
    CHECK(b(0, 1) == Approx(0.0).margin(1e-14));
    CHECK(b(0, 2) == Approx(0.0).margin(1e-14));
}


// ===========================================================================
// Test 11: Patch<T,2>::eval_surface_geometry on twisted z=u*v patch
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_surface_geometry: twisted z=u*v patch",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    std::vector<std::pair<double, double>> test_pts = {
        {0.3, 0.4}, {0.5, 0.5}, {0.8, 0.2}};

    for (auto [u, v] : test_pts) {
        SECTION("at u=" + std::to_string(u) + ", v=" + std::to_string(v)) {
            Eigen::MatrixXd pts(1, 2);
            pts << u, v;
            auto [a1, a2, a3, g_inv, b, jac] = surf.eval_surface_geometry(pts, elem_idx);

            const double D = 1.0 + u * u + v * v;
            const double sqrtD = std::sqrt(D);

            // a_1 = (1, 0, v), a_2 = (0, 1, u)
            CHECK(a1(0, 2) == Approx(v).margin(1e-14));
            CHECK(a2(0, 2) == Approx(u).margin(1e-14));

            // a_3 = (-v, -u, 1)/sqrt(1+u²+v²)
            CHECK(a3(0, 0) == Approx(-v / sqrtD).margin(1e-12));
            CHECK(a3(0, 1) == Approx(-u / sqrtD).margin(1e-12));
            CHECK(a3(0, 2) == Approx( 1.0 / sqrtD).margin(1e-12));
            CHECK(jac(0)   == Approx(sqrtD).margin(1e-12));

            // Inverse metric
            CHECK(g_inv(0, 0) == Approx((1.0 + u * u) / D).margin(1e-12));
            CHECK(g_inv(0, 1) == Approx(-u * v / D).margin(1e-12));
            CHECK(g_inv(0, 2) == Approx((1.0 + v * v) / D).margin(1e-12));

            // Curvature: a_11 = a_22 = 0, a_12 = (0,0,1) → b_11 = b_22 = 0,
            //            b_12 = a_12 · a_3 = 1/sqrt(D)
            CHECK(b(0, 0) == Approx(0.0).margin(1e-14));
            CHECK(b(0, 1) == Approx(1.0 / sqrtD).margin(1e-12));
            CHECK(b(0, 2) == Approx(0.0).margin(1e-14));
        }
    }
}


// ===========================================================================
// Test 12: eval_covariant_derivatives on flat unit square (Christoffels=0)
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_covariant_derivatives: flat unit square",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 0.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    Eigen::MatrixXd pts(1, 2);
    pts << 0.4, 0.6;

    auto r = surf.eval_covariant_derivatives(pts, elem_idx, 2);
    REQUIRE(r.size() == 6);
    const auto& N    = r[0];
    const auto& N_d1 = r[1];
    const auto& N_d2 = r[2];
    const auto& N_d11 = r[3];
    const auto& N_d12 = r[4];
    const auto& N_d22 = r[5];

    // Partition of unity
    CHECK(N.row(0).sum()     == Approx(1.0).margin(1e-14));
    CHECK(N_d1.row(0).sum()  == Approx(0.0).margin(1e-14));
    CHECK(N_d2.row(0).sum()  == Approx(0.0).margin(1e-14));
    CHECK(N_d11.row(0).sum() == Approx(0.0).margin(1e-14));
    CHECK(N_d12.row(0).sum() == Approx(0.0).margin(1e-14));
    CHECK(N_d22.row(0).sum() == Approx(0.0).margin(1e-14));

    // Bilinear: parametric N_{,uu} and N_{,vv} are zero, and Christoffels
    // vanish on a flat axis-aligned patch → covariant 2nd derivs are zero.
    for (int a = 0; a < 4; ++a) {
        CHECK(N_d11(0, a) == Approx(0.0).margin(1e-14));
        CHECK(N_d22(0, a) == Approx(0.0).margin(1e-14));
    }
}


// ===========================================================================
// Test 13: eval_covariant_derivatives on twisted z=u*v patch
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_covariant_derivatives: twisted z=u*v patch",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    const double u = 0.5, v = 0.5;
    Eigen::MatrixXd pts(1, 2);
    pts << u, v;

    auto r = surf.eval_covariant_derivatives(pts, elem_idx, 2);
    REQUIRE(r.size() == 6);
    const auto& N    = r[0];
    const auto& N_d1 = r[1];
    const auto& N_d2 = r[2];
    const auto& N_d11 = r[3];
    const auto& N_d12 = r[4];
    const auto& N_d22 = r[5];

    // Partition of unity (also true for covariant derivatives).
    CHECK(N.row(0).sum()     == Approx(1.0).margin(1e-14));
    CHECK(N_d1.row(0).sum()  == Approx(0.0).margin(1e-14));
    CHECK(N_d2.row(0).sum()  == Approx(0.0).margin(1e-14));
    CHECK(N_d11.row(0).sum() == Approx(0.0).margin(1e-12));
    CHECK(N_d12.row(0).sum() == Approx(0.0).margin(1e-12));
    CHECK(N_d22.row(0).sum() == Approx(0.0).margin(1e-12));

    // For z=u·v the only non-zero geometry second derivative is a_12 = (0,0,1),
    // so Γ^γ_11 = Γ^γ_22 = 0 (both Christoffels at αβ=11,22 vanish). Combined
    // with bilinear shape functions (parametric N_{,uu} = N_{,vv} = 0):
    //   N_{|11} = N_{|22} = 0 identically.
    for (int a = 0; a < 4; ++a) {
        CHECK(N_d11(0, a) == Approx(0.0).margin(1e-12));
        CHECK(N_d22(0, a) == Approx(0.0).margin(1e-12));
    }

    // Spot-check N_{|12} for the (1-u)(1-v) shape (column index 0 per existing
    // test convention) at u=v=0.5, where D = 1.5, Γ^1_12 = v/D = 1/3,
    // Γ^2_12 = u/D = 1/3, N_{,uv} = +1, N_{,u} = -(1-v) = -0.5,
    // N_{,v} = -(1-u) = -0.5:
    //   N_{|12} = 1 - (1/3)(-0.5) - (1/3)(-0.5) = 1 + 1/3 = 4/3.
    CHECK(N_d12(0, 0) == Approx(4.0 / 3.0).margin(1e-12));
}


// ===========================================================================
// Test 14: 3rd-order covariant derivatives on flat bilinear patch
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_covariant_derivatives order=3: flat bilinear",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 0.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    Eigen::MatrixXd pts(1, 2);
    pts << 0.4, 0.6;

    auto r = surf.eval_covariant_derivatives(pts, elem_idx, 3);
    REQUIRE(r.size() == 12);

    // Bilinear → all parametric 3rd derivs zero. Flat axis-aligned →
    // Christoffels and their derivatives all vanish. So all six third-order
    // covariant components are identically zero.
    for (int idx = 6; idx < 12; ++idx) {
        for (int a = 0; a < 4; ++a) {
            CHECK(r[idx](0, a) == Approx(0.0).margin(1e-14));
        }
    }
}


// ===========================================================================
// Test 15: 3rd-order covariant derivatives on flat quadratic identity patch
// (Christoffels=0 → covariant 3rd reduces to parametric 3rd; layout sanity)
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_covariant_derivatives order=3: flat p=2 identity patch",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 0, 1, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(2, kv);
    auto basis_v = std::make_shared<BSpline<double>>(2, kv);

    // Identity map: x = u, y = v, z = 0. Greville-interpolating control net.
    Eigen::MatrixXd cp(9, 3);
    int row = 0;
    for (double v : {0.0, 0.5, 1.0}) {
        for (double u : {0.0, 0.5, 1.0}) {
            cp.row(row++) << u, v, 0.0;
        }
    }
    Patch<double, 2> surf(basis_u, basis_v, cp);

    // Single non-zero element for p=2 with knots {0,0,0,1,1,1}: span_u = span_v = 2,
    // intervals = 5 in each direction → flat = 2*5 + 2 = 12 (decoded as u-inner).
    Index elem_idx = 12;

    Eigen::MatrixXd pts(1, 2);
    pts << 0.3, 0.7;

    auto r = surf.eval_covariant_derivatives(pts, elem_idx, 3);
    REQUIRE(r.size() == 12);

    const Index K = r[0].cols();
    REQUIRE(K == 9);

    // Partition of unity at every order.
    CHECK(r[0].row(0).sum() == Approx(1.0).margin(1e-12));
    for (int idx = 1; idx < 12; ++idx) {
        CHECK(r[idx].row(0).sum() == Approx(0.0).margin(1e-10));
    }

    // p=2 → ∂³/∂u³ ≡ 0 and ∂³/∂v³ ≡ 0 for any tensor-product shape. With
    // Christoffels = 0 these collapse to result[6] (N_{|11,1}) = 0 and
    // result[11] (N_{|22,2}) = 0.
    for (Index a = 0; a < K; ++a) {
        CHECK(r[6](0, a)  == Approx(0.0).margin(1e-12));
        CHECK(r[11](0, a) == Approx(0.0).margin(1e-12));
    }

    // On a flat axis-aligned identity patch, parametric derivatives commute,
    // so result[7] (N_{|11,2} = N_{,uuv}) and result[8] (N_{|12,1} = N_{,uuv})
    // must be equal — the layout is internally consistent on flat patches.
    for (Index a = 0; a < K; ++a) {
        CHECK(r[7](0, a) == Approx(r[8](0, a)).margin(1e-12));
        CHECK(r[9](0, a) == Approx(r[10](0, a)).margin(1e-12));
    }
}


// ===========================================================================
// Test 16: 3rd-order covariant derivatives on twisted z=u*v patch
// (curved geometry — Riemann curvature breaks (γ,α) symmetry)
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_covariant_derivatives order=3: twisted z=u*v patch",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    Eigen::MatrixXd pts(1, 2);
    pts << 0.5, 0.5;

    auto r = surf.eval_covariant_derivatives(pts, elem_idx, 3);
    REQUIRE(r.size() == 12);

    // Partition of unity for all 6 third-order components.
    for (int idx = 6; idx < 12; ++idx) {
        CHECK(r[idx].row(0).sum() == Approx(0.0).margin(1e-12));
    }

    // For z=u·v, only a_{12} = (0,0,1) is non-zero among 2nd geometry derivs;
    // all 3rd geometry derivs vanish. Consequently Γ^δ_{11} = Γ^δ_{22} = 0
    // and Γ^δ_{αβ,γ} for αβ ∈ {11,22} all vanish. Combined with bilinear N
    // (parametric 3rd derivs = 0) this gives:
    //   N_{|11,1} = 0   and   N_{|22,2} = 0
    // for every shape function.
    for (int a = 0; a < 4; ++a) {
        CHECK(r[6](0, a)  == Approx(0.0).margin(1e-12));
        CHECK(r[11](0, a) == Approx(0.0).margin(1e-12));
    }

    // Spot checks for shape 0 = (1-u)(1-v) at u=v=0.5 (analytical from
    // hand-derivation):
    //   N_{|12}   = +4/3
    //   N_{|11,2} = -2 · Γ^2_{12} · N_{|12} = -2·(1/3)·(4/3) = -8/9
    //   N_{|22,1} = -2 · Γ^1_{12} · N_{|12} = -8/9 (u↔v symmetric)
    //   N_{|12,1} = N_{|12,2} = -2/3
    // Note N_{|11,2} ≠ N_{|12,1} on this curved patch — Riemann curvature
    // breaks the full symmetry that holds on flat patches.
    CHECK(r[7](0, 0)  == Approx(-8.0 / 9.0).margin(1e-12));
    CHECK(r[10](0, 0) == Approx(-8.0 / 9.0).margin(1e-12));
    CHECK(r[8](0, 0)  == Approx(-2.0 / 3.0).margin(1e-12));
    CHECK(r[9](0, 0)  == Approx(-2.0 / 3.0).margin(1e-12));

    // The asymmetry: r[7] (N_{|11,2}) ≠ r[8] (N_{|12,1}) on this curved patch.
    CHECK(std::abs(r[7](0, 0) - r[8](0, 0)) > 0.1);
}


// ===========================================================================
// Test 17: order parameter dispatch — sizes match expected layout
// ===========================================================================

TEST_CASE("Patch<double, 2>::eval_covariant_derivatives: order parameter sizes",
          "[geometry][surface][shell-kernel]") {

    auto kv = KnotVector<double>({0, 0, 1, 1});
    auto basis_u = std::make_shared<BSpline<double>>(1, kv);
    auto basis_v = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 0.0;
    Patch<double, 2> surf(basis_u, basis_v, cp);
    Index elem_idx = 4;

    Eigen::MatrixXd pts(1, 2);
    pts << 0.4, 0.6;

    CHECK(surf.eval_covariant_derivatives(pts, elem_idx, 1).size() == 3);
    CHECK(surf.eval_covariant_derivatives(pts, elem_idx, 2).size() == 6);
    CHECK(surf.eval_covariant_derivatives(pts, elem_idx, 3).size() == 12);
    // Default = 2.
    CHECK(surf.eval_covariant_derivatives(pts, elem_idx).size() == 6);
}
