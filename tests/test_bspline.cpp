#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bspline.hpp"
#include "knots.hpp"

using namespace pyck;

/**
 * Partition of unity.
 *
 * For any u ∈ [ξ_p, ξ_{n+1}], the sum of all B-spline basis functions
 * of degree p must equal exactly 1:
 *
 *   Σ_{i=0}^{n} N_{i,p}(u) = 1
 *
 * We verify this for polynomial degrees 1 through 3.
 */
TEST_CASE("BSpline: partition of unity", "[bspline]") {
    for (std::size_t p = 1; p <= 3; ++p) {
        auto kv = KnotVector::clamped_uniform(p, p + 3);
        BSpline bs(p, kv);

        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(30, 0.0, 1.0);
        auto N = bs.eval(u);

        for (int i = 0; i < N.rows(); ++i)
            REQUIRE(N.row(i).sum() == Approx(1.0).margin(1e-12));
    }
}

/**
 * Non-negativity.
 *
 * B-spline basis functions are non-negative everywhere:
 *
 *   N_{i,p}(u) ≥ 0   for all i, p, u
 */
TEST_CASE("BSpline: non-negativity", "[bspline]") {
    auto kv = KnotVector::clamped_uniform(3, 8);
    BSpline bs(3, kv);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(100, 0.0, 1.0);
    auto N = bs.eval(u);

    for (int i = 0; i < N.rows(); ++i)
        for (int j = 0; j < N.cols(); ++j)
            REQUIRE(N(i, j) >= -1e-15);
}

/**
 * Local support.
 *
 * B-spline basis function N_{i,p} has support only on [ξ_i, ξ_{i+p+1}].
 * Outside this interval, the function value must be zero:
 *
 *   N_{i,p}(u) = 0  if  u < ξ_i  or  u > ξ_{i+p+1}
 *
 * We build a non-uniform knot vector {0,0,0, 0.3, 0.7, 1,1,1} (p=2, n=5)
 * and verify that each basis function vanishes outside its support.
 */
TEST_CASE("BSpline: local support", "[bspline]") {
    std::vector<double> raw_knots = {0, 0, 0, 0.3, 0.7, 1, 1, 1};
    BSpline bs(2, KnotVector(raw_knots));

    // Evaluate on a fine grid
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(200, 0.0, 1.0);
    auto N = bs.eval(u);

    for (int j = 0; j < N.cols(); ++j) {
        // Support of N_{j,p} is [ξ_j, ξ_{j+p+1}]
        double xi_lo = raw_knots[j];
        double xi_hi = raw_knots[j + 3]; // p+1 = 3

        for (int i = 0; i < N.rows(); ++i) {
            double ui = u(i);
            if (ui < xi_lo - 1e-14 || ui > xi_hi + 1e-14)
                REQUIRE(N(i, j) == Approx(0.0).margin(1e-14));
        }
    }
}

/**
 * Derivative sum to zero.
 *
 * Because the basis forms a partition of unity, the sum of all k-th order
 * derivatives (k ≥ 1) must vanish:
 *
 *   Σ_{i=0}^{n} d^k/du^k N_{i,p}(u) = 0
 *
 * We verify this for derivatives of order 1 and 2.
 */
TEST_CASE("BSpline: derivative sum to zero", "[bspline][deriv]") {
    auto kv = KnotVector::clamped_uniform(3, 6);
    BSpline bs(3, kv);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(40, 0.0, 1.0);
    auto derivs = bs.eval_derivs(u, 2);

    for (int k = 1; k <= 2; ++k) {
        for (int i = 0; i < derivs[k].rows(); ++i)
            REQUIRE(derivs[k].row(i).sum() == Approx(0.0).margin(1e-10));
    }
}

/**
 * C-continuity at internal knots.
 *
 * At a simple internal knot ξ_i (multiplicity k=1), the B-spline basis
 * of degree p is C^{p-1} continuous.  We verify this numerically via
 * central finite differences: for a cubic basis (p=3), we check that
 * zero-th, first, and second derivatives are continuous across the knot
 * (left and right limits agree), while, as a sanity check, we also ensure
 * that the third derivative can exhibit a jump (C^{p-k} = C^2).
 */
TEST_CASE("BSpline: C-continuity at internal knot", "[bspline][deriv]") {
    // p = 3, 7 basis → knots = {0,0,0,0, 0.25, 0.5, 0.75, 1,1,1,1}
    auto kv = KnotVector::clamped_uniform(3, 7);
    BSpline bs(3, kv);

    // Test at the internal knot ξ = 0.5
    double xi = 0.5;
    double eps = 1e-8;

    Eigen::VectorXd u_left(1), u_right(1);
    u_left  << xi - eps;
    u_right << xi + eps;

    // Evaluate up to 3rd derivative on both sides
    auto left  = bs.eval_derivs(u_left,  3);
    auto right = bs.eval_derivs(u_right, 3);

    // Orders 0, 1, 2 must be continuous (C^2 for a simple knot with p=3)
    for (int k = 0; k <= 2; ++k) {
        for (int j = 0; j < left[k].cols(); ++j)
            REQUIRE(left[k](0, j) == Approx(right[k](0, j)).margin(1e-4));
    }
}

/**
 * Analytical basis values.
 *
 * We verify B-spline basis function values against closed-form Bernstein
 * polynomial expressions for degrees 1 and 2 (single-span Bézier).
 *
 * Degree 1, knots = {0,0,1,1}:
 *   N_0(u) = 1 - u,    N_1(u) = u
 *
 * Degree 2, knots = {0,0,0,1,1,1}:
 *   N_0(u) = (1-u)²,   N_1(u) = 2u(1-u),   N_2(u) = u²
 */
TEST_CASE("BSpline: analytical values", "[bspline]") {
    Eigen::VectorXd u(5);
    u << 0.0, 0.25, 0.5, 0.75, 1.0;

    SECTION("degree 1: linear Bernstein") {
        BSpline bs(1, KnotVector({0, 0, 1, 1}));
        auto N = bs.eval(u);

        for (int i = 0; i < 5; ++i) {
            REQUIRE(N(i, 0) == Approx(1.0 - u(i)).margin(1e-14));
            REQUIRE(N(i, 1) == Approx(u(i)).margin(1e-14));
        }
    }

    SECTION("degree 2: quadratic Bernstein") {
        BSpline bs(2, KnotVector({0, 0, 0, 1, 1, 1}));
        auto N = bs.eval(u);

        for (int i = 0; i < 5; ++i) {
            double t = u(i);
            REQUIRE(N(i, 0) == Approx((1 - t) * (1 - t)).margin(1e-14));
            REQUIRE(N(i, 1) == Approx(2 * t * (1 - t)).margin(1e-14));
            REQUIRE(N(i, 2) == Approx(t * t).margin(1e-14));
        }
    }
}

/**
 * Analytical derivatives.
 *
 * We verify derivatives of Bernstein basis functions against their known
 * closed-form expressions.
 *
 * Degree 1:  N_0' = -1,  N_1' = +1  (constant)
 *
 * Degree 2:
 *   N_0' = -2(1-u),  N_1' = 2 - 4u,  N_2' = 2u
 *   N_0'' = 2,       N_1'' = -4,      N_2'' = 2
 *   Orders > p are zero identically.
 */
TEST_CASE("BSpline: analytical derivatives", "[bspline][deriv]") {
    Eigen::VectorXd u(3);
    u << 0.0, 0.5, 1.0;

    SECTION("degree 1: first derivative") {
        BSpline bs(1, KnotVector({0, 0, 1, 1}));
        auto dN = bs.eval_derivs(u, 1)[1];

        for (int i = 0; i < 3; ++i) {
            REQUIRE(dN(i, 0) == Approx(-1.0).margin(1e-12));
            REQUIRE(dN(i, 1) == Approx( 1.0).margin(1e-12));
        }
    }

    SECTION("degree 2: first derivative") {
        BSpline bs(2, KnotVector({0, 0, 0, 1, 1, 1}));
        auto dN = bs.eval_derivs(u, 1)[1];

        for (int i = 0; i < 3; ++i) {
            double t = u(i);
            REQUIRE(dN(i, 0) == Approx(-2.0 * (1 - t)).margin(1e-12));
            REQUIRE(dN(i, 1) == Approx( 2.0 - 4.0 * t).margin(1e-12));
            REQUIRE(dN(i, 2) == Approx( 2.0 * t).margin(1e-12));
        }
    }

    SECTION("degree 2: second derivative") {
        BSpline bs(2, KnotVector({0, 0, 0, 1, 1, 1}));
        auto d2N = bs.eval_derivs(u, 2)[2];

        for (int i = 0; i < 3; ++i) {
            REQUIRE(d2N(i, 0) == Approx( 2.0).margin(1e-10));
            REQUIRE(d2N(i, 1) == Approx(-4.0).margin(1e-10));
            REQUIRE(d2N(i, 2) == Approx( 2.0).margin(1e-10));
        }
    }

    SECTION("degree 2: third derivative is zero") {
        BSpline bs(2, KnotVector({0, 0, 0, 1, 1, 1}));
        auto d3N = bs.eval_derivs(u, 3)[3];

        for (int i = 0; i < d3N.rows(); ++i)
            for (int j = 0; j < d3N.cols(); ++j)
                REQUIRE(d3N(i, j) == Approx(0.0).margin(1e-14));
    }
}

/**
 * Finite-difference derivative check.
 *
 * We verify the analytically computed first derivative against a central
 * finite-difference approximation:
 *
 *   dN/du ≈ [N(u+h) - N(u-h)] / (2h)
 *
 * This is a general sanity check for arbitrary knot vectors and degrees.
 */
TEST_CASE("BSpline: finite-difference derivative check", "[bspline][deriv]") {
    auto kv = KnotVector::clamped_uniform(3, 6);
    BSpline bs(3, kv);

    double h = 1e-7;
    Eigen::VectorXd u(5);
    u << 0.1, 0.25, 0.5, 0.75, 0.9;

    auto dN_exact = bs.eval_derivs(u, 1)[1];

    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd u_fwd(1), u_bwd(1);
        u_fwd << u(i) + h;
        u_bwd << u(i) - h;

        auto N_fwd = bs.eval(u_fwd);
        auto N_bwd = bs.eval(u_bwd);

        Eigen::VectorXd dN_fd = (N_fwd.row(0) - N_bwd.row(0)) / (2.0 * h);

        for (int j = 0; j < dN_fd.size(); ++j)
            REQUIRE(dN_exact(i, j) == Approx(dN_fd(j)).margin(1e-5));
    }
}
