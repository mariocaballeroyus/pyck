#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bspline.hpp"
#include "surface.hpp"

using namespace pyck;

/**
 * Helper function for clamped uniform knot vector generation
 */
static std::vector<double> clamped_uniform_knots(std::size_t degree,
                                                  std::size_t num_basis)
{
    std::size_t n = num_basis + degree + 1;
    std::vector<double> knots(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (i <= degree)
            knots[i] = 0.0;
        else if (i >= num_basis)
            knots[i] = 1.0;
        else
            knots[i] = static_cast<double>(i - degree)
                      / static_cast<double>(num_basis - degree);
    }
    return knots;
}

/**
 * Helper function to compute Greville abscissae for a B-spline basis
 */
static std::vector<double> greville_abscissae(const BSpline& bs)
{
    std::size_t n = bs.num_basis();
    std::size_t p = bs.degree();
    const auto& T = bs.knots();
    std::vector<double> xi(n);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 1; j <= p; ++j)
            sum += T[i + j];
        xi[i] = sum / static_cast<double>(p);
    }
    return xi;
}

/**
 * Helper: build a flat-plate surface on [0, L_u] x [0, L_v] in 3D.
 * S(u,v) = (L_u * u, L_v * v, 0). Control points placed at Greville
 * abscissae so the linear map is represented exactly.
 */
static SurfacePatch make_flat_plate(BSpline& bu, BSpline& bv,
                                    double L_u, double L_v)
{
    auto xi_u = greville_abscissae(bu);
    auto xi_v = greville_abscissae(bv);
    std::size_t n_u = bu.num_basis();
    std::size_t n_v = bv.num_basis();
    Eigen::MatrixXd P(n_u * n_v, 3);

    for (std::size_t a = 0; a < n_u; ++a) {
        for (std::size_t b = 0; b < n_v; ++b) {
            P.row(a * n_v + b) << L_u * xi_u[a], L_v * xi_v[b], 0.0;
        }
    }

    return SurfacePatch(3, &bu, &bv, P);
}

/**
 * Test the constructor and accessors of the SurfacePatch class
 */
TEST_CASE("SurfacePatch: construction", "[surface]") {
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    BSpline bu(2, ku);
    BSpline bv(2, kv);

    auto surf = make_flat_plate(bu, bv, 2.0, 1.0);

    REQUIRE(surf.gdim() == 3);
    REQUIRE(surf.tdim() == 2);
    REQUIRE(surf.control_points().rows() == 4 * 3);
    REQUIRE(surf.control_points().cols() == 3);
    REQUIRE(surf.basis(0).num_basis() == 4);
    REQUIRE(surf.basis(1).num_basis() == 3);
}

/**
 * Test that eval() returns vectors of matrices of the correct sizes
 */
TEST_CASE("SurfacePatch: eval structure", "[surface]") {
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    BSpline bu(2, ku);
    BSpline bv(2, kv);
    auto surf = make_flat_plate(bu, bv, 1.0, 1.0);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(4, 0.0, 1.0);

    SECTION("order 0") {
        auto result = surf.eval(u, v, 0);
        REQUIRE(result.size() == 1);       // i = 0
        REQUIRE(result[0].size() == 1);    // j = 0
        REQUIRE(result[0][0].rows() == 20); // 5 * 4
        REQUIRE(result[0][0].cols() == 3);
    }

    SECTION("order 1") {
        auto result = surf.eval(u, v, 1);
        REQUIRE(result.size() == 2);
        REQUIRE(result[0].size() == 2);    // j = 0, 1
        REQUIRE(result[1].size() == 1);    // j = 0
    }

    SECTION("order 2") {
        auto result = surf.eval(u, v, 2);
        REQUIRE(result.size() == 3);
        REQUIRE(result[0].size() == 3);    // j = 0, 1, 2
        REQUIRE(result[1].size() == 2);    // j = 0, 1
        REQUIRE(result[2].size() == 1);    // j = 0
    }
}

/**
 * Test flat-plate position: S(u,v) = (L_u*u, L_v*v, 0)
 */
TEST_CASE("SurfacePatch: flat plate position", "[surface]") {
    double L_u = 3.0, L_v = 2.0;
    auto ku = clamped_uniform_knots(1, 2); // linear
    auto kv = clamped_uniform_knots(1, 2);
    BSpline bu(1, ku);
    BSpline bv(1, kv);
    auto surf = make_flat_plate(bu, bv, L_u, L_v);

    Eigen::VectorXd u(3), v(4);
    u << 0.0, 0.5, 1.0;
    v << 0.0, 0.25, 0.75, 1.0;

    auto result = surf.eval(u, v, 0);
    auto& X = result[0][0]; // (12 x 3)

    REQUIRE(X.rows() == 12);

    for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 4; ++q) {
            int row = p * 4 + q;
            REQUIRE(X(row, 0) == Approx(L_u * u(p)).margin(1e-12));
            REQUIRE(X(row, 1) == Approx(L_v * v(q)).margin(1e-12));
            REQUIRE(X(row, 2) == Approx(0.0).margin(1e-12));
        }
    }
}

/**
 * Test flat-plate first derivatives:
 * ∂S/∂u = (L_u, 0, 0),  ∂S/∂v = (0, L_v, 0)
 */
TEST_CASE("SurfacePatch: flat plate first derivatives", "[surface][deriv]") {
    double L_u = 3.0, L_v = 2.0;
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    BSpline bu(2, ku);
    BSpline bv(2, kv);
    auto surf = make_flat_plate(bu, bv, L_u, L_v);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(6, 0.0, 1.0);
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);

    auto result = surf.eval(u, v, 1);
    auto& dSdu = result[1][0]; // ∂S/∂u
    auto& dSdv = result[0][1]; // ∂S/∂v

    int Q = 6 * 5;
    REQUIRE(dSdu.rows() == Q);
    REQUIRE(dSdv.rows() == Q);

    for (int i = 0; i < Q; ++i) {
        // ∂S/∂u = (L_u, 0, 0)
        REQUIRE(dSdu(i, 0) == Approx(L_u).margin(1e-10));
        REQUIRE(dSdu(i, 1) == Approx(0.0).margin(1e-10));
        REQUIRE(dSdu(i, 2) == Approx(0.0).margin(1e-10));

        // ∂S/∂v = (0, L_v, 0)
        REQUIRE(dSdv(i, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(dSdv(i, 1) == Approx(L_v).margin(1e-10));
        REQUIRE(dSdv(i, 2) == Approx(0.0).margin(1e-10));
    }
}

/**
 * Test flat-plate second derivatives: all zero for a linear mapping
 */
TEST_CASE("SurfacePatch: flat plate second derivatives", "[surface][deriv]") {
    double L_u = 3.0, L_v = 2.0;
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    BSpline bu(2, ku);
    BSpline bv(2, kv);
    auto surf = make_flat_plate(bu, bv, L_u, L_v);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(4, 0.0, 1.0);

    auto result = surf.eval(u, v, 2);
    auto& d2Sduu = result[2][0]; // ∂²S/∂u²
    auto& d2Sduv = result[1][1]; // ∂²S/∂u∂v
    auto& d2Sdvv = result[0][2]; // ∂²S/∂v²

    int Q = 5 * 4;
    for (int i = 0; i < Q; ++i) {
        for (int d = 0; d < 3; ++d) {
            REQUIRE(d2Sduu(i, d) == Approx(0.0).margin(1e-10));
            REQUIRE(d2Sduv(i, d) == Approx(0.0).margin(1e-10));
            REQUIRE(d2Sdvv(i, d) == Approx(0.0).margin(1e-10));
        }
    }
}

/**
 * Test parabolic surface S(u,v) = (u, v, u^2 + v^2): position and
 * derivatives up to order 2 against analytical values
 */
TEST_CASE("SurfacePatch: parabolic surface", "[surface][deriv]") {
    // degree 2, 3 basis: {0,0,0,1,1,1}
    // Bernstein basis on [0,1]: N_0=(1-t)^2, N_1=2t(1-t), N_2=t^2
    // To represent f(t)=t^2, control points for z must satisfy:
    //   c0*N_0 + c1*N_1 + c2*N_2 = t^2
    //   → (c0=0, c1=0, c2=1)
    // To represent f(t)=t: c0=0, c1=0.5, c2=1

    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, knots);
    BSpline bv(2, knots);

    std::size_t n_u = 3, n_v = 3;
    Eigen::MatrixXd P(n_u * n_v, 3);

    // Control points for S(u,v) = (u, v, u^2 + v^2)
    // x(u) = u  → Bernstein ctrl: 0, 0.5, 1
    // y(v) = v  → Bernstein ctrl: 0, 0.5, 1
    // z(u,v) = u^2 + v^2 → z_{a,b} = z_u[a] + z_v[b]
    //   where z_u: (0, 0, 1), z_v: (0, 0, 1)
    double x_ctrl[] = {0.0, 0.5, 1.0};
    double y_ctrl[] = {0.0, 0.5, 1.0};
    double zu_ctrl[] = {0.0, 0.0, 1.0};
    double zv_ctrl[] = {0.0, 0.0, 1.0};

    for (std::size_t a = 0; a < n_u; ++a) {
        for (std::size_t b = 0; b < n_v; ++b) {
            P.row(a * n_v + b) << x_ctrl[a], y_ctrl[b], zu_ctrl[a] + zv_ctrl[b];
        }
    }

    SurfacePatch surf(3, &bu, &bv, P);

    Eigen::VectorXd u(5), v(4);
    u << 0.0, 0.25, 0.5, 0.75, 1.0;
    v << 0.0, 0.3, 0.7, 1.0;

    auto result = surf.eval(u, v, 2);

    // --- Check position: S = (u, v, u^2+v^2) ---
    auto& X = result[0][0];
    for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 4; ++q) {
            int row = p * 4 + q;
            double uu = u(p), vv = v(q);
            REQUIRE(X(row, 0) == Approx(uu).margin(1e-12));
            REQUIRE(X(row, 1) == Approx(vv).margin(1e-12));
            REQUIRE(X(row, 2) == Approx(uu*uu + vv*vv).margin(1e-12));
        }
    }

    // --- Check ∂S/∂u = (1, 0, 2u) ---
    auto& dSdu = result[1][0];
    for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 4; ++q) {
            int row = p * 4 + q;
            REQUIRE(dSdu(row, 0) == Approx(1.0).margin(1e-10));
            REQUIRE(dSdu(row, 1) == Approx(0.0).margin(1e-10));
            REQUIRE(dSdu(row, 2) == Approx(2.0 * u(p)).margin(1e-10));
        }
    }

    // --- Check ∂S/∂v = (0, 1, 2v) ---
    auto& dSdv = result[0][1];
    for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 4; ++q) {
            int row = p * 4 + q;
            REQUIRE(dSdv(row, 0) == Approx(0.0).margin(1e-10));
            REQUIRE(dSdv(row, 1) == Approx(1.0).margin(1e-10));
            REQUIRE(dSdv(row, 2) == Approx(2.0 * v(q)).margin(1e-10));
        }
    }

    // --- Check ∂²S/∂u² = (0, 0, 2) ---
    auto& d2Sduu = result[2][0];
    for (int i = 0; i < d2Sduu.rows(); ++i) {
        REQUIRE(d2Sduu(i, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sduu(i, 1) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sduu(i, 2) == Approx(2.0).margin(1e-10));
    }

    // --- Check ∂²S/∂v² = (0, 0, 2) ---
    auto& d2Sdvv = result[0][2];
    for (int i = 0; i < d2Sdvv.rows(); ++i) {
        REQUIRE(d2Sdvv(i, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sdvv(i, 1) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sdvv(i, 2) == Approx(2.0).margin(1e-10));
    }

    // --- Check ∂²S/∂u∂v = (0, 0, 0)  (no cross term in u^2+v^2) ---
    auto& d2Sduv = result[1][1];
    for (int i = 0; i < d2Sduv.rows(); ++i) {
        REQUIRE(d2Sduv(i, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sduv(i, 1) == Approx(0.0).margin(1e-10));
        REQUIRE(d2Sduv(i, 2) == Approx(0.0).margin(1e-10));
    }
}

/**
 * Verify first derivatives against central finite differences
 */
TEST_CASE("SurfacePatch: finite difference derivative check", "[surface][deriv]") {
    auto ku = clamped_uniform_knots(3, 6);
    auto kv = clamped_uniform_knots(3, 5);
    BSpline bu(3, ku);
    BSpline bv(3, kv);

    // Random-ish control points in 3D
    std::size_t n_u = bu.num_basis();
    std::size_t n_v = bv.num_basis();
    Eigen::MatrixXd P(n_u * n_v, 3);
    // Use a deterministic "random" pattern
    for (std::size_t a = 0; a < n_u; ++a) {
        for (std::size_t b = 0; b < n_v; ++b) {
            double s = static_cast<double>(a * n_v + b);
            P.row(a * n_v + b) << std::sin(s) * 2.0,
                                   std::cos(s * 0.7) * 3.0,
                                   std::sin(s * 1.3) * 1.5;
        }
    }

    SurfacePatch surf(3, &bu, &bv, P);

    double h = 1e-7;
    Eigen::VectorXd u(1), v(1);
    // Test at an interior point
    u << 0.4;
    v << 0.6;

    auto result = surf.eval(u, v, 1);
    auto& dSdu = result[1][0]; // (1 x 3)
    auto& dSdv = result[0][1]; // (1 x 3)

    // Finite difference for ∂S/∂u
    Eigen::VectorXd u_fwd(1), u_bwd(1);
    u_fwd << 0.4 + h;
    u_bwd << 0.4 - h;
    auto X_fwd = surf.eval(u_fwd, v, 0)[0][0]; // (1 x 3)
    auto X_bwd = surf.eval(u_bwd, v, 0)[0][0];
    Eigen::RowVectorXd dSdu_fd = (X_fwd.row(0) - X_bwd.row(0)) / (2.0 * h);

    for (int d = 0; d < 3; ++d)
        REQUIRE(dSdu(0, d) == Approx(dSdu_fd(d)).margin(1e-5));

    // Finite difference for ∂S/∂v
    Eigen::VectorXd v_fwd(1), v_bwd(1);
    v_fwd << 0.6 + h;
    v_bwd << 0.6 - h;
    auto Y_fwd = surf.eval(u, v_fwd, 0)[0][0];
    auto Y_bwd = surf.eval(u, v_bwd, 0)[0][0];
    Eigen::RowVectorXd dSdv_fd = (Y_fwd.row(0) - Y_bwd.row(0)) / (2.0 * h);

    for (int d = 0; d < 3; ++d)
        REQUIRE(dSdv(0, d) == Approx(dSdv_fd(d)).margin(1e-5));
}

/**
 * Test that the four corners of the patch interpolate the corner control points
 */
TEST_CASE("SurfacePatch: corner interpolation", "[surface]") {
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    BSpline bu(2, ku);
    BSpline bv(2, kv);
    auto surf = make_flat_plate(bu, bv, 5.0, 3.0);

    Eigen::VectorXd u(2), v(2);
    u << 0.0, 1.0;
    v << 0.0, 1.0;

    auto X = surf.eval(u, v, 0)[0][0]; // 4 x 3

    // Corner (0,0) → (0, 0, 0)
    REQUIRE(X(0, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(X(0, 1) == Approx(0.0).margin(1e-12));

    // Corner (0,1) → (0, 3, 0)
    REQUIRE(X(1, 0) == Approx(0.0).margin(1e-12));
    REQUIRE(X(1, 1) == Approx(3.0).margin(1e-12));

    // Corner (1,0) → (5, 0, 0)
    REQUIRE(X(2, 0) == Approx(5.0).margin(1e-12));
    REQUIRE(X(2, 1) == Approx(0.0).margin(1e-12));

    // Corner (1,1) → (5, 3, 0)
    REQUIRE(X(3, 0) == Approx(5.0).margin(1e-12));
    REQUIRE(X(3, 1) == Approx(3.0).margin(1e-12));
}

/**
 * Test jacobian for a flat plate: S(u,v)=(Lx*u, Ly*v, 0)
 * J = [[Lx, 0], [0, Ly], [0, 0]],  det = Lx*Ly
 */
TEST_CASE("SurfacePatch: jacobian flat plate", "[surface][jacobian]") {
    double Lx = 3.0, Ly = 2.0;
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    BSpline bu(2, ku);
    BSpline bv(2, kv);
    auto surf = make_flat_plate(bu, bv, Lx, Ly);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(4, 0.0, 1.0);
    int Q = 5 * 4;

    auto [jacs, dets] = surf.jacobian(u, v);

    REQUIRE(jacs.size() == static_cast<std::size_t>(Q));
    REQUIRE(dets.size() == Q);

    for (int q = 0; q < Q; ++q) {
        auto& J = jacs[q];
        REQUIRE(J.rows() == 3);
        REQUIRE(J.cols() == 2);

        // col 0 = dS/du = (Lx, 0, 0)
        REQUIRE(J(0, 0) == Approx(Lx).margin(1e-10));
        REQUIRE(J(1, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(J(2, 0) == Approx(0.0).margin(1e-10));

        // col 1 = dS/dv = (0, Ly, 0)
        REQUIRE(J(0, 1) == Approx(0.0).margin(1e-10));
        REQUIRE(J(1, 1) == Approx(Ly).margin(1e-10));
        REQUIRE(J(2, 1) == Approx(0.0).margin(1e-10));

        // det = sqrt(det(J^T J)) = Lx * Ly
        REQUIRE(dets(q) == Approx(Lx * Ly).margin(1e-10));
    }
}

/**
 * Test jacobian for the parabolic surface S(u,v)=(u, v, u²+v²).
 * J.col(0) = (1, 0, 2u), J.col(1) = (0, 1, 2v).
 * det = sqrt(1+4u²+4v²).
 */
TEST_CASE("SurfacePatch: jacobian parabolic surface", "[surface][jacobian]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bu(2, knots);
    BSpline bv(2, knots);

    double x_ctrl[] = {0.0, 0.5, 1.0};
    double y_ctrl[] = {0.0, 0.5, 1.0};
    double zu_ctrl[] = {0.0, 0.0, 1.0};
    double zv_ctrl[] = {0.0, 0.0, 1.0};

    Eigen::MatrixXd P(9, 3);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            P.row(a * 3 + b) << x_ctrl[a], y_ctrl[b], zu_ctrl[a] + zv_ctrl[b];

    SurfacePatch surf(3, &bu, &bv, P);

    Eigen::VectorXd u(5), v(4);
    u << 0.0, 0.25, 0.5, 0.75, 1.0;
    v << 0.0, 0.3, 0.7, 1.0;

    auto [jacs, dets] = surf.jacobian(u, v);

    for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 4; ++q) {
            int idx = p * 4 + q;
            double uu = u(p), vv = v(q);
            auto& J = jacs[idx];

            // J.col(0) = (1, 0, 2u)
            REQUIRE(J(0, 0) == Approx(1.0).margin(1e-10));
            REQUIRE(J(1, 0) == Approx(0.0).margin(1e-10));
            REQUIRE(J(2, 0) == Approx(2.0 * uu).margin(1e-10));

            // J.col(1) = (0, 1, 2v)
            REQUIRE(J(0, 1) == Approx(0.0).margin(1e-10));
            REQUIRE(J(1, 1) == Approx(1.0).margin(1e-10));
            REQUIRE(J(2, 1) == Approx(2.0 * vv).margin(1e-10));

            // det J^T J = (1+4u²)(1+4v²)  (since J^T J is diagonal)
            // Wait: J^T J = [[1+4u², 4uv], [4uv, 1+4v²]]
            //   det = (1+4u²)(1+4v²) - 16u²v² = 1 + 4u² + 4v²
            double expected_det = std::sqrt(1.0 + 4*uu*uu + 4*vv*vv);
            REQUIRE(dets(idx) == Approx(expected_det).margin(1e-10));
        }
    }
}

/**
 * Test jacobian on a skewed linear mapping: x=2u+0.3v, y=0.1u+1.5v, z=0.
 * The Jacobian should be constant and the determinant should match
 * the analytical value.
 */
TEST_CASE("SurfacePatch: jacobian skewed mapping", "[surface][jacobian]") {
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    BSpline bu(2, ku);
    BSpline bv(2, kv);

    auto xi_u = greville_abscissae(bu);
    auto xi_v = greville_abscissae(bv);
    std::size_t n_u = bu.num_basis(), n_v = bv.num_basis();
    Eigen::MatrixXd P(n_u * n_v, 3);
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            P.row(a * n_v + b) << 2.0 * xi_u[a] + 0.3 * xi_v[b],
                                   0.1 * xi_u[a] + 1.5 * xi_v[b],
                                   0.0;

    SurfacePatch surf(3, &bu, &bv, P);

    Eigen::VectorXd u(3), v(3);
    u << 0.2, 0.5, 0.8;
    v << 0.3, 0.5, 0.7;

    auto [jacs, dets] = surf.jacobian(u, v);

    // Expected J = [[2.0, 0.3], [0.1, 1.5], [0, 0]]
    // J^T J = [[2²+0.1², 2*0.3+0.1*1.5], [2*0.3+0.1*1.5, 0.3²+1.5²]]
    //       = [[4.01, 0.75], [0.75, 2.34]]
    //   det = 4.01*2.34 - 0.75² = 8.8209
    double expected_det = std::sqrt(4.01 * 2.34 - 0.75 * 0.75);

    for (std::size_t q = 0; q < jacs.size(); ++q) {
        auto& J = jacs[q];
        REQUIRE(J(0, 0) == Approx(2.0).margin(1e-10));
        REQUIRE(J(0, 1) == Approx(0.3).margin(1e-10));
        REQUIRE(J(1, 0) == Approx(0.1).margin(1e-10));
        REQUIRE(J(1, 1) == Approx(1.5).margin(1e-10));
        REQUIRE(J(2, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(J(2, 1) == Approx(0.0).margin(1e-10));

        REQUIRE(dets(q) == Approx(expected_det).margin(1e-10));
    }
}
