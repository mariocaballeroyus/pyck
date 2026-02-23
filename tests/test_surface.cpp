#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>

#include "bspline.hpp"
#include "knots.hpp"
#include "surface.hpp"

using namespace pyck;

/**
 * Helper: build a (M*N × 2) parameter grid from 1D vectors u and v.
 * Row ordering: (u_0,v_0), (u_0,v_1), ..., i.e. v varies fastest.
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

/**
 * Corner interpolation.
 *
 * For a clamped B-spline surface, the parametric corners must interpolate
 * the corresponding corner control points:
 *
 *   S(u_min, v_min) = P_{0,0}
 *   S(u_max, v_max) = P_{n,m}
 */
TEST_CASE("SurfacePatch: corner interpolation", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, 5.0, 3.0);

    Eigen::VectorXd u(2), v(2);
    u << 0.0, 1.0;
    v << 0.0, 1.0;
    auto params = make_grid(u, v);

    auto X = surf.eval(params);  // 4 x 3

    // (0,0) → (0, 0, 0)
    REQUIRE(X(0, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(X(0, 1) == Approx(0.0).margin(1e-12));

    // (0,1) → (0, 3, 0)
    REQUIRE(X(1, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(X(1, 1) == Approx(3.0).margin(1e-12));

    // (1,0) → (5, 0, 0)
    REQUIRE(X(2, 0) == Approx(5.0).margin(1e-12));
    REQUIRE(X(2, 1) == Approx(0.0).margin(1e-12));

    // (1,1) → (5, 3, 0)
    REQUIRE(X(3, 0) == Approx(5.0).margin(1e-12));
    REQUIRE(X(3, 1) == Approx(3.0).margin(1e-12));
}

/**
 * Rigid body transformation (translation invariance).
 *
 * If all control points are translated by a vector T, the evaluated
 * surface must translate by the same amount:
 *
 *   S_new(u,v) = S_old(u,v) + T
 */
TEST_CASE("SurfacePatch: rigid body translation", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf_original = SurfacePatch::flat_plate({&bu, &bv}, 3.0, 2.0);

    // Translate all control points by T = (10, -5, 7)
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

/**
 * Affine covariance (rotation).
 *
 * Rotating all control points by a rotation matrix R must produce
 * the same rotation of the evaluated surface points:
 *
 *   S_new(u,v) = R · S_old(u,v)
 *
 * We use a 90° rotation about the z-axis.
 */
TEST_CASE("SurfacePatch: affine covariance (rotation)", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf_original = SurfacePatch::flat_plate({&bu, &bv}, 3.0, 2.0);

    Eigen::Matrix3d R;
    R << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

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

/**
 * Isoparametric curve test.
 *
 * Fixing v = v* produces a 1D curve S(u, v*) that must equal a 1D
 * B-spline curve whose control points are Q_i = Σ_j M_{j,q}(v*) · P_{i,j}.
 */
TEST_CASE("SurfacePatch: isoparametric curve", "[surface]") {
    BSpline bu(2, KnotVector::clamped_uniform(2, 4));
    BSpline bv(2, KnotVector::clamped_uniform(2, 3));
    auto surf = SurfacePatch::flat_plate({&bu, &bv}, 4.0, 2.0);

    double v_star = 0.4;
    Eigen::VectorXd u_pts = Eigen::VectorXd::LinSpaced(10, 0.0, 1.0);

    // Build params: (10 x 2) with v = v_star
    Eigen::MatrixXd params(10, 2);
    params.col(0) = u_pts;
    params.col(1).setConstant(v_star);

    auto X_iso = surf.eval(params);  // (10 x 3)

    // Compute effective 1D control points: Q_i = Σ_j M_j(v*) · P_{i,j}
    Eigen::VectorXd vv(1);
    vv << v_star;
    auto Mv = bv.eval(vv);  // (1 x n_v)

    std::size_t n_u = bu.num_basis(), n_v = bv.num_basis();
    const auto& P = surf.control_points();
    Eigen::MatrixXd Q(n_u, 3);
    Q.setZero();
    for (std::size_t i = 0; i < n_u; ++i)
        for (std::size_t j = 0; j < n_v; ++j)
            Q.row(i) += Mv(0, j) * P.row(i * n_v + j);

    // Evaluate the 1D B-spline curve
    auto Nu = bu.eval(u_pts);  // (10 x n_u)
    Eigen::MatrixXd X_curve = Nu * Q;  // (10 x 3)

    for (int i = 0; i < X_iso.rows(); ++i)
        for (int d = 0; d < 3; ++d)
            REQUIRE(X_iso(i, d) == Approx(X_curve(i, d)).margin(1e-12));
}

/**
 * Normal vector perpendicularity.
 *
 * The surface normal n = ∂S/∂u × ∂S/∂v must be perpendicular to both
 * tangent vectors.  Tested on the paraboloid S(u,v) = (u, v, u²+v²).
 */
TEST_CASE("SurfacePatch: normal vector perpendicularity", "[surface]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));
    BSpline bv(2, KnotVector(knots));
    auto surf = SurfacePatch::paraboloid({&bu, &bv});

    Eigen::VectorXd u(4), v(4);
    u << 0.1, 0.3, 0.6, 0.9;
    v << 0.2, 0.4, 0.7, 0.8;
    auto params = make_grid(u, v);

    auto result = surf.eval_derivs(params, 1);
    auto& dSdu = result[1][0];
    auto& dSdv = result[0][1];

    for (int i = 0; i < dSdu.rows(); ++i) {
        Eigen::Vector3d tu = dSdu.row(i).transpose();
        Eigen::Vector3d tv = dSdv.row(i).transpose();
        Eigen::Vector3d normal = tu.cross(tv);

        REQUIRE(normal.dot(tu) == Approx(0.0).margin(1e-10));
        REQUIRE(normal.dot(tv) == Approx(0.0).margin(1e-10));
        REQUIRE(normal(2) > 0.0);  // upward-pointing for paraboloid
    }
}

/**
 * Parabolic surface: analytical derivatives.
 *
 * For S(u,v) = (u, v, u²+v²):
 *   ∂S/∂u = (1, 0, 2u),  ∂S/∂v = (0, 1, 2v)
 *   ∂²S/∂u² = (0, 0, 2), ∂²S/∂v² = (0, 0, 2), ∂²S/∂u∂v = 0
 */
TEST_CASE("SurfacePatch: parabolic surface analytical derivatives", "[surface][deriv]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, KnotVector(knots));
    BSpline bv(2, KnotVector(knots));
    auto surf = SurfacePatch::paraboloid({&bu, &bv});

    Eigen::VectorXd u(5), v(4);
    u << 0.0, 0.25, 0.5, 0.75, 1.0;
    v << 0.0, 0.3, 0.7, 1.0;
    auto params = make_grid(u, v);

    auto result = surf.eval_derivs(params, 2);
    auto& X      = result[0][0];
    auto& dSdu   = result[1][0];
    auto& dSdv   = result[0][1];
    auto& d2Sduu = result[2][0];
    auto& d2Sdvv = result[0][2];
    auto& d2Sduv = result[1][1];

    for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 4; ++q) {
            int row = p * 4 + q;
            double uu = u(p), vv = v(q);

            REQUIRE(X(row, 0) == Approx(uu).margin(1e-12));
            REQUIRE(X(row, 1) == Approx(vv).margin(1e-12));
            REQUIRE(X(row, 2) == Approx(uu * uu + vv * vv).margin(1e-12));

            REQUIRE(dSdu(row, 0) == Approx(1.0).margin(1e-10));
            REQUIRE(dSdu(row, 1) == Approx(0.0).margin(1e-10));
            REQUIRE(dSdu(row, 2) == Approx(2.0 * uu).margin(1e-10));

            REQUIRE(dSdv(row, 0) == Approx(0.0).margin(1e-10));
            REQUIRE(dSdv(row, 1) == Approx(1.0).margin(1e-10));
            REQUIRE(dSdv(row, 2) == Approx(2.0 * vv).margin(1e-10));
        }
    }

    for (int i = 0; i < d2Sduu.rows(); ++i) {
        REQUIRE(d2Sduu(i, 2) == Approx(2.0).margin(1e-10));
        REQUIRE(d2Sdvv(i, 2) == Approx(2.0).margin(1e-10));
        REQUIRE(d2Sduv(i, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sduv(i, 1) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sduv(i, 2) == Approx(0.0).margin(1e-10));
    }
}

/**
 * Jacobian against analytical solution (parabolic surface).
 *
 * For S(u,v) = (u, v, u²+v²):
 *   J = [[1, 0], [0, 1], [2u, 2v]]
 *   |J| = √(1 + 4u² + 4v²)
 */
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

    for (int p = 0; p < 5; ++p) {
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

            double expected_det = std::sqrt(1.0 + 4 * uu * uu + 4 * vv * vv);
            REQUIRE(dets(idx) == Approx(expected_det).margin(1e-10));
        }
    }
}

/**
 * Finite-difference derivative check for a general surface.
 *
 * Builds a surface with non-trivial control points and verifies
 * ∂S/∂u and ∂S/∂v against central finite differences.
 */
TEST_CASE("SurfacePatch: finite-difference derivative check", "[surface][deriv]") {
    BSpline bu(3, KnotVector::clamped_uniform(3, 6));
    BSpline bv(3, KnotVector::clamped_uniform(3, 5));

    // Deterministic pseudo-random control points
    std::size_t n_u = bu.num_basis(), n_v = bv.num_basis();
    Eigen::MatrixXd P(n_u * n_v, 3);
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b) {
            double s = static_cast<double>(a * n_v + b);
            P.row(a * n_v + b) << std::sin(s) * 2.0,
                                   std::cos(s * 0.7) * 3.0,
                                   std::sin(s * 1.3) * 1.5;
        }
    SurfacePatch surf(3, {&bu, &bv}, P);

    double h = 1e-7;
    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.6;

    auto result = surf.eval_derivs(params, 1);
    auto& dSdu = result[1][0];
    auto& dSdv = result[0][1];

    // ∂S/∂u via finite differences
    Eigen::MatrixXd p_fwd(1, 2), p_bwd(1, 2);
    p_fwd << 0.4 + h, 0.6;
    p_bwd << 0.4 - h, 0.6;
    auto X_fwd = surf.eval(p_fwd);
    auto X_bwd = surf.eval(p_bwd);
    Eigen::RowVectorXd dSdu_fd = (X_fwd.row(0) - X_bwd.row(0)) / (2.0 * h);

    for (int d = 0; d < 3; ++d)
        REQUIRE(dSdu(0, d) == Approx(dSdu_fd(d)).margin(1e-5));

    // ∂S/∂v via finite differences
    p_fwd << 0.4, 0.6 + h;
    p_bwd << 0.4, 0.6 - h;
    auto Y_fwd = surf.eval(p_fwd);
    auto Y_bwd = surf.eval(p_bwd);
    Eigen::RowVectorXd dSdv_fd = (Y_fwd.row(0) - Y_bwd.row(0)) / (2.0 * h);

    for (int d = 0; d < 3; ++d)
        REQUIRE(dSdv(0, d) == Approx(dSdv_fd(d)).margin(1e-5));
}
