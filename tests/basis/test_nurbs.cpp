#include "catch.hpp"

#include <cmath>
#include <vector>

#include "nurbs.hpp"
#include "knots.hpp"

using namespace pyck;

/**
 * Partition of unity.
 *
 * For any valid u, the rational basis functions sum to one:
 *   Σ_i R_{i,p}(u) = 1
 *
 * This is independent of the (positive) weight choice.
 */
TEST_CASE("NURBS: partition of unity", "[basis][nurbs]") {
    auto kv = clamped_uniform_knots<double>(3, 7);
    NURBS<double> nb(3, kv, {1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 1.0});

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(30, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span = nb.find_span(u(i));
        auto R = nb.eval_on_span(pt, span, 0)[0];
        REQUIRE(R.row(0).sum() == Approx(1.0).margin(1e-12));
    }
}

/**
 * Reduction to B-spline when all weights are equal.
 *
 * If w_i = c (constant) for all i, then R_{i,p} = N_{i,p} and the rational
 * basis collapses to the underlying B-spline.
 */
TEST_CASE("NURBS: equal weights ⇒ B-spline", "[basis][nurbs]") {
    auto kv = clamped_uniform_knots<double>(2, 5);
    BSpline<double> bs(2, kv);
    NURBS<double> nb(2, kv, std::vector<double>(5, 2.7));

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(20, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span_b = bs.find_span(u(i));
        auto span_n = nb.find_span(u(i));
        REQUIRE(span_b == span_n);

        auto N = bs.eval_on_span(pt, span_b, 2);
        auto R = nb.eval_on_span(pt, span_n, 2);
        for (std::size_t k = 0; k < 3; ++k) {
            for (int j = 0; j < N[k].cols(); ++j) {
                REQUIRE(R[k](0, j) == Approx(N[k](0, j)).margin(1e-10));
            }
        }
    }
}

/**
 * Exact representation of a unit circle.
 *
 * The classical 9-point quadratic NURBS circle has
 *   degree     = 2
 *   knots      = {0,0,0, 1/4,1/4, 1/2,1/2, 3/4,3/4, 1,1,1}
 *   weights    = {1, √2/2, 1, √2/2, 1, √2/2, 1, √2/2, 1}
 *   control pts (for unit circle in xy-plane):
 *     ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1), (-1,  0),
 *     (-1, -1), ( 0, -1), ( 1, -1), ( 1,  0)
 *
 * Evaluating x(u) = Σ_i R_i(u) P_i must give a point on the unit circle for
 * every u ∈ [0, 1].
 */
TEST_CASE("NURBS: exact unit circle", "[basis][nurbs]") {
    const double s = std::sqrt(2.0) / 2.0;
    KnotVector<double> kv({0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1});
    NURBS<double> nb(2, kv, {1, s, 1, s, 1, s, 1, s, 1});

    // Control points (x, y); each row is one CP.
    std::vector<std::pair<double, double>> cps = {
        { 1,  0}, { 1,  1}, { 0,  1}, {-1,  1}, {-1,  0},
        {-1, -1}, { 0, -1}, { 1, -1}, { 1,  0},
    };

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(50, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span = nb.find_span(u(i));
        auto R = nb.eval_on_span(pt, span, 0)[0]; // (1 × 3)

        double x = 0.0, y = 0.0;
        for (int j = 0; j <= 2; ++j) {
            std::size_t global = span - 2 + j;
            x += R(0, j) * cps[global].first;
            y += R(0, j) * cps[global].second;
        }
        REQUIRE(x * x + y * y == Approx(1.0).margin(1e-12));
    }
}

/**
 * Derivatives via finite differences.
 *
 * Compare the analytical 1st and 2nd derivatives produced by NURBS::eval_all
 * against centered finite differences on R(u) = w_i N_i(u) / W(u).
 *
 * eval_all uses a global indexing of the n basis functions, so finite
 * differences across an internal knot still compare the same R_i.
 */
TEST_CASE("NURBS: derivatives match finite differences", "[basis][nurbs]") {
    auto kv = clamped_uniform_knots<double>(3, 7);
    std::vector<double> w = {1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 1.0};
    NURBS<double> nb(3, kv, w);

    const double h = 1e-6;
    const Eigen::Index n = static_cast<Eigen::Index>(nb.num_basis());
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(15, 0.05, 0.95);
    for (int i = 0; i < u.size(); ++i) {
        const double u0 = u(i);
        Eigen::VectorXd p0(1), pm(1), pp(1);
        p0 << u0; pm << u0 - h; pp << u0 + h;

        auto R0 = nb.eval_all(p0, 2);                   // [R, R', R''] (1 × n)
        auto R_minus = nb.eval_all(pm, 0)[0];           // (1 × n)
        auto R_plus  = nb.eval_all(pp, 0)[0];           // (1 × n)
        auto R_pm2   = nb.eval_all(p0, 1);              // for 2nd-order FD on R'

        for (Eigen::Index j = 0; j < n; ++j) {
            const double fd1 = (R_plus(0, j) - R_minus(0, j)) / (2.0 * h);
            REQUIRE(R0[1](0, j) == Approx(fd1).margin(1e-6));
        }

        // 2nd derivative FD: (R(u+h) − 2 R(u) + R(u−h)) / h²
        for (Eigen::Index j = 0; j < n; ++j) {
            const double fd2 = (R_plus(0, j) - 2.0 * R0[0](0, j) + R_minus(0, j))
                             / (h * h);
            REQUIRE(R0[2](0, j) == Approx(fd2).margin(1e-3));
        }
    }
}
