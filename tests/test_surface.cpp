#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>
#include <string>

#include "bspline.hpp"
#include "knots.hpp"
#include "surface.hpp"

using namespace pyck;

/**
 * Helper: build a (M*N × 2) parameter grid from 1D vectors u and v.
 */
static Eigen::MatrixXd make_grid(const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v)
{
    int M = u.size(), N = v.size();
    Eigen::MatrixXd params(M * N, 2);
    for (int p = 0; p < M; ++p)
        for (int q = 0; q < N; ++q) {
            params(p * N + q, 0) = u(p);
            params(p * N + q, 1) = v(q);
        }
    return params;
}

// =============================================================================
// eval() tests — unchanged
// =============================================================================

TEST_CASE("SurfacePatch: corner interpolation", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, 5.0, 3.0);

    Eigen::VectorXd u(2), v(2);
    u << 0.0, 1.0;
    v << 0.0, 1.0;
    auto params = make_grid(u, v);
    auto X = surf.eval(params);

    REQUIRE(X(0, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(X(0, 1) == Approx(0.0).margin(1e-12));
    REQUIRE(X(1, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(X(1, 1) == Approx(3.0).margin(1e-12));
    REQUIRE(X(2, 0) == Approx(5.0).margin(1e-12));
    REQUIRE(X(2, 1) == Approx(0.0).margin(1e-12));
    REQUIRE(X(3, 0) == Approx(5.0).margin(1e-12));
    REQUIRE(X(3, 1) == Approx(3.0).margin(1e-12));
}

TEST_CASE("SurfacePatch: rigid body translation", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf_original = SurfacePatch::flat_plate({&bu, &bv}, 3.0, 2.0);

    Eigen::RowVector3d T(10.0, -5.0, 7.0);
    Eigen::MatrixXd P_translated = surf_original.control_points().rowwise() + T;
    SurfacePatch surf_translated(3, {&bu, &bv}, P_translated);

    auto params = make_grid(Eigen::VectorXd::LinSpaced(5, 0.0, 1.0),
                            Eigen::VectorXd::LinSpaced(4, 0.0, 1.0));
    auto X_old = surf_original.eval(params);
    auto X_new = surf_translated.eval(params);

    for (int i = 0; i < X_old.rows(); ++i)
        for (int d = 0; d < 3; ++d)
            REQUIRE(X_new(i, d) == Approx(X_old(i, d) + T(d)).margin(1e-12));
}

TEST_CASE("SurfacePatch: affine covariance (rotation)", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf_original = SurfacePatch::flat_plate({&bu, &bv}, 3.0, 2.0);

    Eigen::Matrix3d R;
    R << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    Eigen::MatrixXd P_rotated = (R * surf_original.control_points().transpose()).transpose();
    SurfacePatch surf_rotated(3, {&bu, &bv}, P_rotated);

    auto params = make_grid(Eigen::VectorXd::LinSpaced(5, 0.0, 1.0),
                            Eigen::VectorXd::LinSpaced(4, 0.0, 1.0));
    auto X_old = surf_original.eval(params);
    auto X_new = surf_rotated.eval(params);

    for (int i = 0; i < X_old.rows(); ++i) {
        Eigen::Vector3d expected = R * X_old.row(i).transpose();
        for (int d = 0; d < 3; ++d)
            REQUIRE(X_new(i, d) == Approx(expected(d)).margin(1e-12));
    }
}

TEST_CASE("SurfacePatch: isoparametric curve", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, 4.0, 2.0);

    double v_star = 0.4;
    Eigen::VectorXd u_pts = Eigen::VectorXd::LinSpaced(10, 0.0, 1.0);
    Eigen::MatrixXd params(10, 2);
    params.col(0) = u_pts;
    params.col(1).setConstant(v_star);
    auto X_iso = surf.eval(params);

    Eigen::VectorXd vv(1);
    vv << v_star;
    auto Mv = bv.eval(vv);

    std::size_t n_u = bu.num_basis(), n_v = bv.num_basis();
    const auto& P = surf.control_points();
    Eigen::MatrixXd Q(n_u, 3);
    Q.setZero();
    for (std::size_t i = 0; i < n_u; ++i)
        for (std::size_t j = 0; j < n_v; ++j)
            Q.row(i) += Mv(0, j) * P.row(i * n_v + j);

    auto Nu = bu.eval(u_pts);
    Eigen::MatrixXd X_curve = Nu * Q;
    for (int i = 0; i < X_iso.rows(); ++i)
        for (int d = 0; d < 3; ++d)
            REQUIRE(X_iso(i, d) == Approx(X_curve(i, d)).margin(1e-12));
}

// =============================================================================
// Jacobian tests — unchanged
// =============================================================================

TEST_CASE("SurfacePatch: jacobian parabolic surface", "[surface][jacobian]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));
    BSpline bv(2, KnotVector(knots));
    auto surf = SurfacePatch::paraboloid({&bu, &bv});

    Eigen::VectorXd u(5), v(4);
    u << 0.0, 0.25, 0.5, 0.75, 1.0;
    v << 0.0, 0.3, 0.7, 1.0;
    auto params = make_grid(u, v);
    auto jacs = surf.jacobian(params);
    auto dets = surf.jacobian_det(params);

    for (int p = 0; p < 5; ++p)
        for (int q = 0; q < 4; ++q) {
            int idx = p * 4 + q;
            double uu = u(p), vv = v(q);
            auto& J = jacs[idx];
            REQUIRE(J(0, 0) == Approx(1.0).margin(1e-10));
            REQUIRE(J(1, 0) == Approx(0.0).margin(1e-10));
            REQUIRE(J(2, 0) == Approx(2.0 * uu).margin(1e-10));
            REQUIRE(J(0, 1) == Approx(0.0).margin(1e-10));
            REQUIRE(J(1, 1) == Approx(1.0).margin(1e-10));
            REQUIRE(J(2, 1) == Approx(2.0 * vv).margin(1e-10));
            REQUIRE(dets(idx) == Approx(std::sqrt(1+4*uu*uu+4*vv*vv)).margin(1e-10));
        }
}

// =============================================================================
// eval_physical_derivs — local tangent-plane tests
// =============================================================================

/**
 * Key check: verify the expected keys are present for each order.
 * Local surface coords → only x, y (2 tangent directions).
 */
TEST_CASE("SurfacePatch: physical derivs key check", "[surface][phys_deriv]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));
    BSpline bv(2, KnotVector(knots));
    auto surf = SurfacePatch::paraboloid({&bu, &bv});

    Eigen::MatrixXd params(1, 2);
    params << 0.5, 0.5;

    SECTION("Order 1") {
        auto r = surf.eval_physical_derivs(params, 1);
        REQUIRE(r.count("N") == 1);
        REQUIRE(r.count("dNdx") == 1);
        REQUIRE(r.count("dNdy") == 1);
        REQUIRE(r.count("dNdz") == 0);  // no z in local tangent frame
        REQUIRE(r.count("dNdxdx") == 0);
    }

    SECTION("Order 2") {
        auto r = surf.eval_physical_derivs(params, 2);
        REQUIRE(r.count("dNdxdx") == 1);
        REQUIRE(r.count("dNdxdy") == 1);
        REQUIRE(r.count("dNdydy") == 1);
        REQUIRE(r.count("dNdxdz") == 0); // no z
    }

    SECTION("Order 3") {
        auto r = surf.eval_physical_derivs(params, 3);
        REQUIRE(r.count("dNdxdxdx") == 1);
        REQUIRE(r.count("dNdxdxdy") == 1);
        REQUIRE(r.count("dNdxdydy") == 1);
        REQUIRE(r.count("dNdydydy") == 1);
        REQUIRE(r.size() == 1 + 2 + 3 + 4);  // N + 2 first + 3 second + 4 third = 10
    }
}

/**
 * Partition of unity: Σ N_A = 1,  so Σ dN_A/dx_i = 0 for all i.
 */
TEST_CASE("SurfacePatch: partition of unity for physical derivs", "[surface][phys_deriv]") {
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, 4.0, 2.0);

    Eigen::VectorXd u(3), v(3);
    u << 0.2, 0.5, 0.8;
    v << 0.3, 0.6, 0.9;
    auto params = make_grid(u, v);

    auto r = surf.eval_physical_derivs(params, 2);

    for (int q = 0; q < params.rows(); ++q) {
        // Σ N_A = 1
        REQUIRE(r.at("N").row(q).sum() == Approx(1.0).margin(1e-12));
        // Σ dN_A/dx = 0
        REQUIRE(r.at("dNdx").row(q).sum() == Approx(0.0).margin(1e-10));
        REQUIRE(r.at("dNdy").row(q).sum() == Approx(0.0).margin(1e-10));
        // Σ d²N_A/dxdx = 0
        REQUIRE(r.at("dNdxdx").row(q).sum() == Approx(0.0).margin(1e-10));
        REQUIRE(r.at("dNdxdy").row(q).sum() == Approx(0.0).margin(1e-10));
        REQUIRE(r.at("dNdydy").row(q).sum() == Approx(0.0).margin(1e-10));
    }
}

/**
 * Flat plate: S(u,v) = (Lu·u, Lv·v, 0).
 *
 * The local frame is: e1 = (1,0,0), e2 = (0,1,0).
 * J_local = diag(Lu, Lv), so:
 *   dN/dx = (1/Lu) dN/du,  dN/dy = (1/Lv) dN/dv
 */
TEST_CASE("SurfacePatch: flat plate 1st-order local derivs", "[surface][phys_deriv]") {
    double Lu = 4.0, Lv = 2.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, Lu, Lv);

    Eigen::VectorXd u(3), v(3);
    u << 0.2, 0.5, 0.8;
    v << 0.3, 0.6, 0.9;
    auto params = make_grid(u, v);

    auto r = surf.eval_physical_derivs(params, 1);

    // Compare dN/dx against (1/Lu) * dN/du
    auto dNdu = bu.eval_derivs(u, 1)[1];  // (3 × n_u)
    auto Nv   = bv.eval(v);               // (3 × n_v)
    auto Nu   = bu.eval(u);               // (3 × n_u)
    auto dNdv = bv.eval_derivs(v, 1)[1];  // (3 × n_v)

    std::size_t n_u = bu.num_basis(), n_v = bv.num_basis();

    for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) {
            int row = p * 3 + q;
            for (std::size_t A = 0; A < n_u; ++A) {
                for (std::size_t B = 0; B < n_v; ++B) {
                    std::size_t col = A * n_v + B;
                    double expected_dx = (1.0 / Lu) * dNdu(p, A) * Nv(q, B);
                    double expected_dy = (1.0 / Lv) * Nu(p, A) * dNdv(q, B);
                    REQUIRE(r.at("dNdx")(row, col) == Approx(expected_dx).margin(1e-10));
                    REQUIRE(r.at("dNdy")(row, col) == Approx(expected_dy).margin(1e-10));
                }
            }
        }
    }
}

/**
 * Flat plate: 2nd-order local derivs.
 *
 * d²N/dxdx = (1/Lu²) d²N/du², d²N/dxdy = (1/(Lu·Lv)) d²N/dudv,
 * d²N/dydy = (1/Lv²) d²N/dv²
 */
TEST_CASE("SurfacePatch: flat plate 2nd-order local derivs", "[surface][phys_deriv]") {
    double Lu = 4.0, Lv = 2.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, Lu, Lv);

    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.6;

    auto r = surf.eval_physical_derivs(params, 2);

    Eigen::VectorXd uu(1), vv(1);
    uu << 0.4; vv << 0.6;

    auto bu_d = bu.eval_derivs(uu, 2);  // [N, dN/du, d²N/du²]
    auto bv_d = bv.eval_derivs(vv, 2);

    std::size_t n_u = bu.num_basis(), n_v = bv.num_basis();

    for (std::size_t A = 0; A < n_u; ++A) {
        for (std::size_t B = 0; B < n_v; ++B) {
            std::size_t col = A * n_v + B;
            double exp_dxdx = (1.0/(Lu*Lu)) * bu_d[2](0, A) * bv_d[0](0, B);
            double exp_dxdy = (1.0/(Lu*Lv)) * bu_d[1](0, A) * bv_d[1](0, B);
            double exp_dydy = (1.0/(Lv*Lv)) * bu_d[0](0, A) * bv_d[2](0, B);
            REQUIRE(r.at("dNdxdx")(0, col) == Approx(exp_dxdx).margin(1e-10));
            REQUIRE(r.at("dNdxdy")(0, col) == Approx(exp_dxdy).margin(1e-10));
            REQUIRE(r.at("dNdydy")(0, col) == Approx(exp_dydy).margin(1e-10));
        }
    }
}

/**
 * Finite-difference check for 1st-order physical derivs on flat plate.
 *
 * On a flat plate, local x maps to global x (e1 along u), so we can
 * perturb in parametric space: dx_local = Lu * du.
 */
TEST_CASE("SurfacePatch: flat plate FD 1st-order", "[surface][phys_deriv]") {
    double Lu = 4.0, Lv = 2.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, Lu, Lv);

    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.6;
    const auto& P = surf.control_points();

    auto r = surf.eval_physical_derivs(params, 1);

    double h = 1e-7;
    // ∂S/∂x_local ≈ (S(u+h/Lu,v) - S(u-h/Lu,v)) / (2h)
    Eigen::MatrixXd pf(1,2), pb(1,2);
    pf << 0.4 + h/Lu, 0.6;
    pb << 0.4 - h/Lu, 0.6;
    Eigen::RowVectorXd dSdx_fd = (surf.eval(pf) - surf.eval(pb)).row(0) / (2.0*h);
    Eigen::RowVectorXd dSdx_api = (r.at("dNdx") * P).row(0);
    for (int d = 0; d < 3; ++d)
        REQUIRE(dSdx_api(d) == Approx(dSdx_fd(d)).margin(1e-5));

    // ∂S/∂y_local ≈ (S(u,v+h/Lv) - S(u,v-h/Lv)) / (2h)
    pf << 0.4, 0.6 + h/Lv;
    pb << 0.4, 0.6 - h/Lv;
    Eigen::RowVectorXd dSdy_fd = (surf.eval(pf) - surf.eval(pb)).row(0) / (2.0*h);
    Eigen::RowVectorXd dSdy_api = (r.at("dNdy") * P).row(0);
    for (int d = 0; d < 3; ++d)
        REQUIRE(dSdy_api(d) == Approx(dSdy_fd(d)).margin(1e-5));
}

/**
 * FD check for 2nd-order physical derivs.
 *
 * d²N/dxdx ≈ (dNdx(x+h) - dNdx(x-h)) / (2h)
 */
TEST_CASE("SurfacePatch: flat plate FD 2nd-order", "[surface][phys_deriv]") {
    double Lu = 4.0, Lv = 2.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, Lu, Lv);

    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.6;
    auto r0 = surf.eval_physical_derivs(params, 2);

    double h = 1e-6;
    Eigen::MatrixXd pf(1,2), pb(1,2);

    // d²N/dxdx ≈ Δ(dNdx) / (2h) with x-perturbation
    pf << 0.4 + h/Lu, 0.6;
    pb << 0.4 - h/Lu, 0.6;
    auto rf = surf.eval_physical_derivs(pf, 1);
    auto rb = surf.eval_physical_derivs(pb, 1);
    auto d2Ndxdx_fd = (rf.at("dNdx") - rb.at("dNdx")) / (2.0 * h);
    for (int c = 0; c < r0.at("dNdxdx").cols(); ++c)
        REQUIRE(r0.at("dNdxdx")(0,c) == Approx(d2Ndxdx_fd(0,c)).margin(1e-3));

    // d²N/dxdy ≈ Δ(dNdx) / (2h) with y-perturbation
    pf << 0.4, 0.6 + h/Lv;
    pb << 0.4, 0.6 - h/Lv;
    rf = surf.eval_physical_derivs(pf, 1);
    rb = surf.eval_physical_derivs(pb, 1);
    auto d2Ndxdy_fd = (rf.at("dNdx") - rb.at("dNdx")) / (2.0 * h);
    for (int c = 0; c < r0.at("dNdxdy").cols(); ++c)
        REQUIRE(r0.at("dNdxdy")(0,c) == Approx(d2Ndxdy_fd(0,c)).margin(1e-3));
}

/**
 * 3rd-order physical derivatives: partition of unity and geometry check.
 *
 * For any order k, Σ_A d^k N_A / dx_{i₁}...dx_{iₖ} = 0
 * (derivative of the constant partition-of-unity 1).
 *
 * On a flat plate (linear geometry), the 3rd-order geometry derivatives
 * d³S/dx³ = Σ d³N_A/dx³ * P_A should be zero because S is linear.
 */
TEST_CASE("SurfacePatch: 3rd-order partition of unity & geometry", "[surface][phys_deriv]") {
    double Lu = 3.0, Lv = 2.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, Lu, Lv);

    Eigen::VectorXd u(3), v(3);
    u << 0.2, 0.5, 0.8;
    v << 0.3, 0.6, 0.9;
    auto params = make_grid(u, v);

    auto r = surf.eval_physical_derivs(params, 3);
    const auto& P = surf.control_points();

    for (const auto& key : {"dNdxdxdx", "dNdxdxdy", "dNdxdydy", "dNdydydy"}) {
        REQUIRE(r.count(key) == 1);
        auto& m = r.at(key);
        for (int q = 0; q < m.rows(); ++q) {
            // Partition of unity: sum of 3rd-order derivs = 0
            REQUIRE(m.row(q).sum() == Approx(0.0).margin(1e-8));
            // Geometry 3rd deriv = 0 (linear surface)
            Eigen::RowVectorXd geom_deriv = m.row(q) * P;
            for (int d = 0; d < geom_deriv.cols(); ++d)
                REQUIRE(geom_deriv(d) == Approx(0.0).margin(1e-8));
        }
    }
}

/**
 * Physical derivative assembly test: membrane strain operator B_m.
 *
 * For a flat plate with S = (Lu*u, Lv*v, 0), the membrane strain
 * operator B_m maps nodal displacements to in-plane strains:
 *   ε_xx = Σ dN_A/dx · u_{A,x}
 *   ε_yy = Σ dN_A/dy · u_{A,y}
 *   ε_xy = Σ (dN_A/dy · u_{A,x} + dN_A/dx · u_{A,y}) / 2
 *
 * For a uniform displacement u=(c,0,0), ε must be zero (rigid body).
 * For a linear displacement u=(x,0,0), ε_xx = 1.
 */
TEST_CASE("SurfacePatch: membrane strain assembly (flat plate)", "[surface][phys_deriv][assembly]") {
    double Lu = 4.0, Lv = 2.0;
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, Lu, Lv);

    Eigen::VectorXd u(3), v(3);
    u << 0.2, 0.5, 0.8;
    v << 0.3, 0.6, 0.9;
    auto params = make_grid(u, v);

    auto r = surf.eval_physical_derivs(params, 1);
    const auto& N = r.at("N");
    const auto& dNdx = r.at("dNdx");
    const auto& dNdy = r.at("dNdy");
    const auto& P = surf.control_points();  // (n × 3)
    int n = P.rows();

    // Rigid body: uniform displacement d = (1, 0, 0) at every control point
    Eigen::VectorXd d_rigid = Eigen::VectorXd::Zero(n * 3);
    for (int A = 0; A < n; ++A)
        d_rigid(A * 3 + 0) = 1.0;  // u_x = 1 for all

    for (int q = 0; q < params.rows(); ++q) {
        double eps_xx = 0.0, eps_yy = 0.0, eps_xy = 0.0;
        for (int A = 0; A < n; ++A) {
            double ux = d_rigid(A * 3 + 0), uy = d_rigid(A * 3 + 1);
            eps_xx += dNdx(q, A) * ux;
            eps_yy += dNdy(q, A) * uy;
            eps_xy += 0.5 * (dNdy(q, A) * ux + dNdx(q, A) * uy);
        }
        REQUIRE(eps_xx == Approx(0.0).margin(1e-10));
        REQUIRE(eps_yy == Approx(0.0).margin(1e-10));
        REQUIRE(eps_xy == Approx(0.0).margin(1e-10));
    }

    // Linear stretch in x: d = (x_phys, 0, 0) = (Lu*u, 0, 0)
    // Physical position x = N * P, so x_A = P(A, 0)
    // ε_xx = Σ dN_A/dx * P(A,0) = 1  (since dS_x/dx = 1 for identity mapping in x)
    for (int q = 0; q < params.rows(); ++q) {
        double eps_xx = 0.0;
        for (int A = 0; A < n; ++A)
            eps_xx += dNdx(q, A) * P(A, 0);
        REQUIRE(eps_xx == Approx(1.0).margin(1e-10));
    }
}

/**
 * Physical derivative assembly test: bending strain operator (Kirchhoff-Love).
 *
 * For a flat plate, the bending strains are:
 *   κ_xx = Σ d²N_A/dx² · w_A
 *   κ_yy = Σ d²N_A/dy² · w_A
 *   κ_xy = 2 Σ d²N_A/dxdy · w_A
 *
 * where w_A is the out-of-plane displacement at control point A.
 *
 * For w = constant, all κ = 0 (rigid body).
 * For w = x² / 2, κ_xx = 1, κ_yy = 0, κ_xy = 0.
 */
TEST_CASE("SurfacePatch: bending strain assembly (flat plate)", "[surface][phys_deriv][assembly]") {
    double Lu = 4.0, Lv = 2.0;
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, Lu, Lv);

    Eigen::VectorXd u(3), v(3);
    u << 0.2, 0.5, 0.8;
    v << 0.3, 0.6, 0.9;
    auto params = make_grid(u, v);

    auto r = surf.eval_physical_derivs(params, 2);
    const auto& d2Ndxdx = r.at("dNdxdx");
    const auto& d2Ndydy = r.at("dNdydy");
    const auto& d2Ndxdy = r.at("dNdxdy");
    const auto& P = surf.control_points();
    int n = P.rows();

    // Rigid body: constant w = 1
    Eigen::VectorXd w_rigid = Eigen::VectorXd::Ones(n);
    for (int q = 0; q < params.rows(); ++q) {
        double kappa_xx = d2Ndxdx.row(q).dot(w_rigid);
        double kappa_yy = d2Ndydy.row(q).dot(w_rigid);
        double kappa_xy = 2.0 * d2Ndxdy.row(q).dot(w_rigid);
        REQUIRE(kappa_xx == Approx(0.0).margin(1e-10));
        REQUIRE(kappa_yy == Approx(0.0).margin(1e-10));
        REQUIRE(kappa_xy == Approx(0.0).margin(1e-10));
    }

    // Quadratic deflection: w(x,y) = x²/2, so κ_xx = d²w/dx² = 1.
    // On a flat plate x_A = P(A,0) are the physical x-coordinates.
    // For a cubic basis, the quadratic field is exactly representable,
    // so we compute the B-spline control values by L2-projection:
    //   w_A = (N^T N)^{-1} N^T w_phys
    // evaluated at the nodes (Greville abscissae would be exact too).
    //
    // Simpler exact approach: the x-coord is exactly represented by the
    // basis (x = N * P[:,0]), so x² is representable piecewise by degree
    // 2p. For degree p ≥ 2, w = x²/2 is exactly Σ N_A * w_A where
    // w_A = P(A,0)²/2  ONLY if the basis can exactly reproduce x².
    // For B-splines on a clamped-uniform knot vector, the Greville
    // control points coincide with the x-coordinate values, so
    // x = Σ N_A * x_A exactly. But x² ≠ Σ N_A * x_A² in general.
    //
    // So instead, we solve for the exact control values via L2 projection
    // at many sample points.
    int n_samp = 50;
    Eigen::VectorXd u_samp = Eigen::VectorXd::LinSpaced(n_samp, 0.01, 0.99);
    Eigen::VectorXd v_samp = Eigen::VectorXd::LinSpaced(n_samp, 0.01, 0.99);
    auto params_samp = make_grid(u_samp, v_samp);
    auto N_samp = r.at("N");  // can't reuse — need to evaluate at sample pts
    // Re-evaluate at sample points
    auto r_samp = surf.eval_physical_derivs(params_samp, 1);
    Eigen::MatrixXd N_s = r_samp.at("N"); // (n_samp² × n)
    // x-coordinates at sample points
    Eigen::VectorXd x_samp = (N_s * P.col(0));  // physical x
    Eigen::VectorXd w_samp_vals = x_samp.array().square() / 2.0;
    // Solve N^T N w = N^T w_samp (least squares)
    Eigen::VectorXd w_quad = (N_s.transpose() * N_s).ldlt().solve(N_s.transpose() * w_samp_vals);

    // Now verify κ
    auto r2 = surf.eval_physical_derivs(params, 2);
    const auto& d2dxdx = r2.at("dNdxdx");
    const auto& d2dydy = r2.at("dNdydy");
    const auto& d2dxdy = r2.at("dNdxdy");

    for (int q = 0; q < params.rows(); ++q) {
        double kappa_xx = d2dxdx.row(q).dot(w_quad);
        double kappa_yy = d2dydy.row(q).dot(w_quad);
        double kappa_xy = 2.0 * d2dxdy.row(q).dot(w_quad);
        REQUIRE(kappa_xx == Approx(1.0).margin(1e-6));
        REQUIRE(kappa_yy == Approx(0.0).margin(1e-6));
        REQUIRE(kappa_xy == Approx(0.0).margin(1e-6));
    }
}

/**
 * Bulge patch analytical benchmark.
 *
 * Bi-quadratic Bernstein surface (knots {0,0,0,1,1,1}) with control points
 * mapping (u,v) → (2u, 2v, 0) except the center control point P_{1,1} = (1,1,1)
 * which creates a bulge. This triggers all Christoffel correction terms.
 *
 * Evaluated at (u=0.25, v=0.5) for basis function R_{1,1} (index 4 in storage).
 *
 * Exact analytical values:
 *   a = √17/2,  b = 0,  d = 2
 *   Jinv = [[2/√17, 0], [0, 1/2]]
 *   dN/dx = 1/√17,  dN/dy = 0
 *   d²N/dx² = -128/289,  d²N/dy² = -6/17,  d²N/dxdy = 0
 */
TEST_CASE("SurfacePatch: bulge patch analytical benchmark", "[surface][phys_deriv][benchmark]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));
    BSpline bv(2, KnotVector(knots));

    // 3×3 = 9 control points, v runs fastest
    // P_{i,j} at row (i*3 + j)
    Eigen::MatrixXd P(9, 3);
    P.row(0) << 0, 0, 0;  // P_{0,0}
    P.row(1) << 0, 1, 0;  // P_{0,1}
    P.row(2) << 0, 2, 0;  // P_{0,2}
    P.row(3) << 1, 0, 0;  // P_{1,0}
    P.row(4) << 1, 1, 1;  // P_{1,1}  ← bulge (z=1)
    P.row(5) << 1, 2, 0;  // P_{1,2}
    P.row(6) << 2, 0, 0;  // P_{2,0}
    P.row(7) << 2, 1, 0;  // P_{2,1}
    P.row(8) << 2, 2, 0;  // P_{2,2}

    SurfacePatch surf(3, {&bu, &bv}, P);

    Eigen::MatrixXd params(1, 2);
    params << 0.25, 0.5;

    // ── Verify surface position ──
    auto X = surf.eval(params);
    REQUIRE(X(0, 0) == Approx(0.5).margin(1e-12));
    REQUIRE(X(0, 1) == Approx(1.0).margin(1e-12));
    REQUIRE(X(0, 2) == Approx(0.1875).margin(1e-12));

    // ── Verify Jacobian ──
    auto jacs = surf.jacobian(params);
    auto& J = jacs[0];
    // g1 = (2, 0, 0.5), g2 = (0, 2, 0)
    REQUIRE(J(0, 0) == Approx(2.0).margin(1e-12));
    REQUIRE(J(1, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(J(2, 0) == Approx(0.5).margin(1e-12));
    REQUIRE(J(0, 1) == Approx(0.0).margin(1e-12));
    REQUIRE(J(1, 1) == Approx(2.0).margin(1e-12));
    REQUIRE(J(2, 1) == Approx(0.0).margin(1e-12));

    // ── Physical derivatives ──
    auto r = surf.eval_physical_derivs(params, 2);

    // Basis function R_{1,1} is at column index 4 (1*3 + 1)
    const int col = 4;

    // -- 0th order: R_{1,1}(0.25, 0.5) = 0.1875 --
    REQUIRE(r.at("N")(0, col) == Approx(0.1875).margin(1e-12));

    // -- 1st order --
    // dN/dx = 1/√17 ≈ 0.242535625
    REQUIRE(r.at("dNdx")(0, col) == Approx(1.0 / std::sqrt(17.0)).margin(1e-10));
    // dN/dy = 0  (v=0.5 is a symmetry line for B_1(v))
    REQUIRE(r.at("dNdy")(0, col) == Approx(0.0).margin(1e-10));

    // -- 2nd order --
    // d²N/dx² = -128/289 ≈ -0.442906574
    REQUIRE(r.at("dNdxdx")(0, col) == Approx(-128.0 / 289.0).margin(1e-10));
    // d²N/dy² = -6/17 ≈ -0.352941176
    REQUIRE(r.at("dNdydy")(0, col) == Approx(-6.0 / 17.0).margin(1e-10));
    // d²N/dxdy = 0
    REQUIRE(r.at("dNdxdy")(0, col) == Approx(0.0).margin(1e-10));
}
