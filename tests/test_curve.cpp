#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>
#include <string>

#include "bspline.hpp"
#include "knots.hpp"
#include "curve.hpp"

using namespace pyck;

// =============================================================================
// eval() tests
// =============================================================================

TEST_CASE("CurvePatch: endpoint interpolation", "[curve]") {
    double L = 5.0;
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    auto curve = CurvePatch::line_segment(&bu, L);

    Eigen::MatrixXd params(2, 1);
    params << 0.0, 1.0;
    auto X = curve.eval(params);

    REQUIRE(X(0, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(X(0, 1) == Approx(0.0).margin(1e-12));
    REQUIRE(X(0, 2) == Approx(0.0).margin(1e-12));
    REQUIRE(X(1, 0) == Approx(L).margin(1e-12));
    REQUIRE(X(1, 1) == Approx(0.0).margin(1e-12));
    REQUIRE(X(1, 2) == Approx(0.0).margin(1e-12));
}

TEST_CASE("CurvePatch: rigid body translation", "[curve]") {
    double L = 3.0;
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    auto curve_original = CurvePatch::line_segment(&bu, L);

    Eigen::RowVector3d T(10.0, -5.0, 7.0);
    Eigen::MatrixXd P_translated = curve_original.control_points().rowwise() + T;
    CurvePatch curve_translated(3, &bu, P_translated);

    Eigen::MatrixXd params = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);
    params.resize(5, 1);
    auto X_old = curve_original.eval(params);
    auto X_new = curve_translated.eval(params);

    for (int i = 0; i < X_old.rows(); ++i)
        for (int d = 0; d < 3; ++d)
            REQUIRE(X_new(i, d) == Approx(X_old(i, d) + T(d)).margin(1e-12));
}

TEST_CASE("CurvePatch: affine covariance (rotation)", "[curve]") {
    double L = 3.0;
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    auto curve_original = CurvePatch::line_segment(&bu, L);

    // 90-degree rotation around z-axis
    Eigen::Matrix3d R;
    R << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    Eigen::MatrixXd P_rotated = (R * curve_original.control_points().transpose()).transpose();
    CurvePatch curve_rotated(3, &bu, P_rotated);

    Eigen::MatrixXd params = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);
    params.resize(5, 1);
    auto X_old = curve_original.eval(params);
    auto X_new = curve_rotated.eval(params);

    for (int i = 0; i < X_old.rows(); ++i) {
        Eigen::Vector3d expected = R * X_old.row(i).transpose();
        for (int d = 0; d < 3; ++d)
            REQUIRE(X_new(i, d) == Approx(expected(d)).margin(1e-12));
    }
}

// =============================================================================
// Jacobian tests
// =============================================================================

TEST_CASE("CurvePatch: jacobian line segment", "[curve][jacobian]") {
    double L = 4.0;
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    auto curve = CurvePatch::line_segment(&bu, L);

    Eigen::MatrixXd params(3, 1);
    params << 0.0, 0.5, 1.0;
    auto jacs = curve.jacobian(params);
    auto dets = curve.jacobian_det(params);

    for (int q = 0; q < 3; ++q) {
        auto& J = jacs[q];
        REQUIRE(J.rows() == 3);
        REQUIRE(J.cols() == 1);
        REQUIRE(J(0, 0) == Approx(L).margin(1e-10));
        REQUIRE(J(1, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(J(2, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(dets(q) == Approx(L).margin(1e-10));
    }
}

TEST_CASE("CurvePatch: jacobian parabolic curve", "[curve][jacobian]") {
    // C(u) = (u, u^2, 0) — quadratic Bernstein basis
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));
    // Control points for parabola C(u) = (u, u², 0):
    // With quadratic Bernstein on [0,1]:
    //   P0 = (0, 0, 0), P1 = (0.5, 0, 0), P2 = (1, 1, 0)
    // Actually C(u) = (1-u)²P0 + 2u(1-u)P1 + u²P2
    // For x(u) = u: P0_x=0, P1_x=0.5, P2_x=1  ✓
    // For y(u) = u²: P0_y=0, P1_y=0, P2_y=1  ✓
    Eigen::MatrixXd P(3, 3);
    P.row(0) << 0.0, 0.0, 0.0;
    P.row(1) << 0.5, 0.0, 0.0;
    P.row(2) << 1.0, 1.0, 0.0;

    CurvePatch curve(3, &bu, P);

    Eigen::MatrixXd params(5, 1);
    params << 0.0, 0.25, 0.5, 0.75, 1.0;
    auto jacs = curve.jacobian(params);
    auto dets = curve.jacobian_det(params);

    for (int q = 0; q < 5; ++q) {
        double u = params(q, 0);
        // dC/du = (1, 2u, 0)
        REQUIRE(jacs[q](0, 0) == Approx(1.0).margin(1e-10));
        REQUIRE(jacs[q](1, 0) == Approx(2.0 * u).margin(1e-10));
        REQUIRE(jacs[q](2, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(dets(q) == Approx(std::sqrt(1.0 + 4.0 * u * u)).margin(1e-10));
    }
}

// =============================================================================
// eval_physical_derivs — arc-length tests
// =============================================================================

TEST_CASE("CurvePatch: physical derivs key check", "[curve][phys_deriv]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));

    Eigen::MatrixXd P(3, 3);
    P.row(0) << 0.0, 0.0, 0.0;
    P.row(1) << 0.5, 0.0, 0.0;
    P.row(2) << 1.0, 1.0, 0.0;
    CurvePatch curve(3, &bu, P);

    Eigen::MatrixXd params(1, 1);
    params << 0.5;

    SECTION("Order 1") {
        auto r = curve.eval_physical_derivs(params, 1);
        REQUIRE(r.count("N") == 1);
        REQUIRE(r.count("dNdx") == 1);
        REQUIRE(r.count("dNdy") == 0);
        REQUIRE(r.count("dNdxdx") == 0);
        REQUIRE(r.size() == 2);
    }

    SECTION("Order 2") {
        auto r = curve.eval_physical_derivs(params, 2);
        REQUIRE(r.count("dNdxdx") == 1);
        REQUIRE(r.count("dNdxdy") == 0);
        REQUIRE(r.size() == 3);  // N + dNdx + dNdxdx
    }

    SECTION("Order 3") {
        auto r = curve.eval_physical_derivs(params, 3);
        REQUIRE(r.count("dNdxdxdx") == 1);
        REQUIRE(r.size() == 4);  // N + dNdx + dNdxdx + dNdxdxdx
    }
}

TEST_CASE("CurvePatch: partition of unity for physical derivs", "[curve][phys_deriv]") {
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    auto curve = CurvePatch::line_segment(&bu, 4.0);

    Eigen::MatrixXd params(5, 1);
    params << 0.1, 0.3, 0.5, 0.7, 0.9;

    auto r = curve.eval_physical_derivs(params, 3);

    for (int q = 0; q < params.rows(); ++q) {
        // Σ N_A = 1
        REQUIRE(r.at("N").row(q).sum() == Approx(1.0).margin(1e-12));
        // Σ dN_A/dx = 0
        REQUIRE(r.at("dNdx").row(q).sum() == Approx(0.0).margin(1e-10));
        // Σ d²N_A/dx² = 0
        REQUIRE(r.at("dNdxdx").row(q).sum() == Approx(0.0).margin(1e-10));
        // Σ d³N_A/dx³ = 0
        REQUIRE(r.at("dNdxdxdx").row(q).sum() == Approx(0.0).margin(1e-10));
    }
}

TEST_CASE("CurvePatch: line segment 1st-order physical derivs", "[curve][phys_deriv]") {
    double L = 4.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    auto curve = CurvePatch::line_segment(&bu, L);

    Eigen::VectorXd u_pts(3);
    u_pts << 0.2, 0.5, 0.8;
    Eigen::MatrixXd params(3, 1);
    params.col(0) = u_pts;

    auto r = curve.eval_physical_derivs(params, 1);

    // Compare dN/dx against (1/L) * dN/du
    auto dNdu = bu.eval_derivs(u_pts, 1)[1];  // (3 × n)

    std::size_t n = bu.num_basis();
    for (int q = 0; q < 3; ++q) {
        for (std::size_t A = 0; A < n; ++A) {
            double expected_dx = (1.0 / L) * dNdu(q, A);
            REQUIRE(r.at("dNdx")(q, A) == Approx(expected_dx).margin(1e-10));
        }
    }
}

TEST_CASE("CurvePatch: line segment 2nd-order physical derivs", "[curve][phys_deriv]") {
    double L = 4.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    auto curve = CurvePatch::line_segment(&bu, L);

    Eigen::VectorXd u_pts(1);
    u_pts << 0.4;
    Eigen::MatrixXd params(1, 1);
    params(0, 0) = 0.4;

    auto r = curve.eval_physical_derivs(params, 2);
    auto bu_d = bu.eval_derivs(u_pts, 2);

    std::size_t n = bu.num_basis();
    for (std::size_t A = 0; A < n; ++A) {
        double exp_dxdx = (1.0 / (L * L)) * bu_d[2](0, A);
        REQUIRE(r.at("dNdxdx")(0, A) == Approx(exp_dxdx).margin(1e-10));
    }
}

TEST_CASE("CurvePatch: line segment FD 1st-order", "[curve][phys_deriv]") {
    double L = 4.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    auto curve = CurvePatch::line_segment(&bu, L);

    Eigen::MatrixXd params(1, 1);
    params << 0.4;
    const auto& P = curve.control_points();

    auto r = curve.eval_physical_derivs(params, 1);

    double h = 1e-7;
    // dC/dx ≈ (C(u + h/L) - C(u - h/L)) / (2h)
    Eigen::MatrixXd pf(1, 1), pb(1, 1);
    pf << 0.4 + h / L;
    pb << 0.4 - h / L;
    Eigen::RowVectorXd dCdx_fd = (curve.eval(pf) - curve.eval(pb)).row(0) / (2.0 * h);
    Eigen::RowVectorXd dCdx_api = (r.at("dNdx") * P).row(0);
    for (int d = 0; d < 3; ++d)
        REQUIRE(dCdx_api(d) == Approx(dCdx_fd(d)).margin(1e-5));
}

TEST_CASE("CurvePatch: line segment FD 2nd-order", "[curve][phys_deriv]") {
    double L = 4.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    auto curve = CurvePatch::line_segment(&bu, L);

    Eigen::MatrixXd params(1, 1);
    params << 0.4;
    auto r0 = curve.eval_physical_derivs(params, 2);

    double h = 1e-6;
    Eigen::MatrixXd pf(1, 1), pb(1, 1);

    // d²N/dxdx ≈ (dNdx(u + h/L) - dNdx(u - h/L)) / (2h)
    pf << 0.4 + h / L;
    pb << 0.4 - h / L;
    auto rf = curve.eval_physical_derivs(pf, 1);
    auto rb = curve.eval_physical_derivs(pb, 1);
    auto d2Ndxdx_fd = (rf.at("dNdx") - rb.at("dNdx")) / (2.0 * h);
    for (int c = 0; c < r0.at("dNdxdx").cols(); ++c)
        REQUIRE(r0.at("dNdxdx")(0, c) == Approx(d2Ndxdx_fd(0, c)).margin(1e-3));
}

/**
 * 3rd-order physical derivatives: partition of unity and geometry check.
 *
 * On a line segment (linear geometry), the 3rd-order geometry derivatives
 * d³C/dx³ = Σ d³N_A/dx³ * P_A should be zero because C is linear.
 */
TEST_CASE("CurvePatch: 3rd-order partition of unity & geometry", "[curve][phys_deriv]") {
    double L = 3.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    auto curve = CurvePatch::line_segment(&bu, L);

    Eigen::MatrixXd params(5, 1);
    params << 0.1, 0.3, 0.5, 0.7, 0.9;

    auto r = curve.eval_physical_derivs(params, 3);
    const auto& P = curve.control_points();

    auto& m = r.at("dNdxdxdx");
    for (int q = 0; q < m.rows(); ++q) {
        // Partition of unity: sum of 3rd-order derivs = 0
        REQUIRE(m.row(q).sum() == Approx(0.0).margin(1e-8));
        // Geometry 3rd deriv = 0 (linear curve)
        Eigen::RowVectorXd geom_deriv = m.row(q) * P;
        for (int d = 0; d < geom_deriv.cols(); ++d)
            REQUIRE(geom_deriv(d) == Approx(0.0).margin(1e-8));
    }
}

/**
 * Curved geometry analytical benchmark.
 *
 * Quadratic Bernstein curve (knots {0,0,0,1,1,1}) with control points
 * P0 = (0,0,0), P1 = (0.5,0,0), P2 = (1,1,0)
 * which maps C(u) = (u, u², 0).
 *
 * At u = 0.5:
 *   g1 = dC/du = (1, 1, 0)
 *   a = ||g1|| = √2
 *   dN/dx = (1/√2) dN/du
 *
 * For basis function N_1(u) = 2u(1-u) (middle Bernstein, index 1):
 *   N_1(0.5) = 0.5
 *   dN_1/du = 2 - 4u = 0 at u=0.5
 *   d²N_1/du² = -4
 *
 *   dN_1/dx = (1/√2) * 0 = 0
 *
 *   da/du = g1 · (d²C/du²) / a = (1,1,0)·(0,2,0) / √2 = 2/√2 = √2
 *   d²N_1/dx² = (1/a²) d²N/du² - (da/du / a³) dN/du
 *             = (1/2)(-4) - (√2 / (2√2)) * 0
 *             = -2
 */
TEST_CASE("CurvePatch: curved geometry analytical benchmark", "[curve][phys_deriv][benchmark]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));

    Eigen::MatrixXd P(3, 3);
    P.row(0) << 0.0, 0.0, 0.0;
    P.row(1) << 0.5, 0.0, 0.0;
    P.row(2) << 1.0, 1.0, 0.0;

    CurvePatch curve(3, &bu, P);

    Eigen::MatrixXd params(1, 1);
    params << 0.5;

    // ── Verify curve position ──
    auto X = curve.eval(params);
    REQUIRE(X(0, 0) == Approx(0.5).margin(1e-12));
    REQUIRE(X(0, 1) == Approx(0.25).margin(1e-12));
    REQUIRE(X(0, 2) == Approx(0.0).margin(1e-12));

    // ── Verify Jacobian ──
    auto jacs = curve.jacobian(params);
    auto& J = jacs[0];
    // g1 = (1, 1, 0)
    REQUIRE(J(0, 0) == Approx(1.0).margin(1e-12));
    REQUIRE(J(1, 0) == Approx(1.0).margin(1e-12));
    REQUIRE(J(2, 0) == Approx(0.0).margin(1e-12));

    // ── Physical derivatives ──
    auto r = curve.eval_physical_derivs(params, 2);

    // Basis function N_1 (middle Bernstein) is at column index 1
    const int col = 1;

    // -- 0th order: N_1(0.5) = 2 * 0.5 * 0.5 = 0.5 --
    REQUIRE(r.at("N")(0, col) == Approx(0.5).margin(1e-12));

    // -- 1st order: dN_1/dx = (1/√2) * 0 = 0 --
    REQUIRE(r.at("dNdx")(0, col) == Approx(0.0).margin(1e-10));

    // -- 2nd order: d²N_1/dx² = -2 --
    REQUIRE(r.at("dNdxdx")(0, col) == Approx(-2.0).margin(1e-10));
}
