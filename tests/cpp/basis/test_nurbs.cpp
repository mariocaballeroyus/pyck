#include "catch.hpp"

#include <cmath>
#include <vector>

#include "nurbs.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "evaluation.hpp"

using namespace pyck;

// Helper: evaluate all n basis functions at m points via eval_on_span loop.
// Returns a vector of (order+1) matrices for derivatives.
static std::vector<Eigen::MatrixXd> eval_all_dense(const Basis<double>& basis,
                                                    const Eigen::VectorXd& pts,
                                                    Index order = 0)
{
    const int n = basis.num_basis();
    const int p = basis.degree();
    const int m = pts.size();

    std::vector<Eigen::MatrixXd> results(order + 1);
    for (Index k = 0; k <= order; ++k)
        results[k] = Eigen::MatrixXd::Zero(m, n);

    Evaluator<double> ev;
    std::vector<Eigen::MatrixXd> local(order + 1);
    for (int i = 0; i < m; ++i) {
        int span = basis.find_span(pts[i]);
        Eigen::VectorXd pt(1); pt << pts[i];
        basis.eval_on_span(pt, span, local, ev);
        // local[k] is (p+1) x 1 col-major (N x Q with Q=1).
        for (Index k = 0; k <= order; ++k) {
            results[k].row(i).segment(span - p, p + 1) = local[k].col(0).transpose();
        }
    }
    return results;
}

/**
 * Partition of unity.
 *
 * For any valid u, the rational basis functions sum to one:
 *   Σ_i R_{i,p}(u) = 1
 *
 * This is independent of the (positive) weight choice.
 */
TEST_CASE("NURBS: partition of unity", "[basis][nurbs]") {
    auto kv = KnotVector<double>::clamped_uniform(3, 7);
    NURBS<double> nb(3, kv,
        (Vector<double>(7) << 1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 1.0).finished());

    Evaluator<double> ev;
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(30, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span = nb.find_span(u(i));
        std::vector<Eigen::MatrixXd> _bufR(1); nb.eval_on_span(pt, span, _bufR, ev); auto R = _bufR[0];
        REQUIRE(R.col(0).sum() == Approx(1.0).margin(1e-12));
    }
}

/**
 * Reduction to B-spline when all weights are equal.
 *
 * If w_i = c (constant) for all i, then R_{i,p} = N_{i,p} and the rational
 * basis collapses to the underlying B-spline.
 */
TEST_CASE("NURBS: equal weights ⇒ B-spline", "[basis][nurbs]") {
    auto kv = KnotVector<double>::clamped_uniform(2, 5);
    BSpline<double> bs(2, kv);
    NURBS<double> nb(2, kv, Vector<double>::Constant(5, 2.7));

    Evaluator<double> ev;
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(20, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span_b = bs.find_span(u(i));
        auto span_n = nb.find_span(u(i));
        REQUIRE(span_b == span_n);

        std::vector<Eigen::MatrixXd> N(3); bs.eval_on_span(pt, span_b, N, ev);
        std::vector<Eigen::MatrixXd> R(3); nb.eval_on_span(pt, span_n, R, ev);
        for (std::size_t k = 0; k < 3; ++k) {
            for (int j = 0; j < N[k].rows(); ++j) {
                REQUIRE(R[k](j, 0) == Approx(N[k](j, 0)).margin(1e-10));
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
    KnotVector<double> kv(std::vector<double>{0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1});
    NURBS<double> nb(2, kv,
        (Vector<double>(9) << 1, s, 1, s, 1, s, 1, s, 1).finished());

    // Control points (x, y); each row is one CP.
    std::vector<std::pair<double, double>> cps = {
        { 1,  0}, { 1,  1}, { 0,  1}, {-1,  1}, {-1,  0},
        {-1, -1}, { 0, -1}, { 1, -1}, { 1,  0},
    };

    Evaluator<double> ev;
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(50, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span = nb.find_span(u(i));
        std::vector<Eigen::MatrixXd> _bufR(1); nb.eval_on_span(pt, span, _bufR, ev); auto R = _bufR[0]; // (3 × 1)

        double x = 0.0, y = 0.0;
        for (int j = 0; j <= 2; ++j) {
            std::size_t global = span - 2 + j;
            x += R(j, 0) * cps[global].first;
            y += R(j, 0) * cps[global].second;
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
    auto kv = KnotVector<double>::clamped_uniform(3, 7);
    Vector<double> w(7);
    w << 1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 1.0;
    NURBS<double> nb(3, kv, w);

    const double h = 1e-6;
    const Eigen::Index n = static_cast<Eigen::Index>(nb.num_basis());
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(15, 0.05, 0.95);
    for (int i = 0; i < u.size(); ++i) {
        const double u0 = u(i);
        Eigen::VectorXd p0(1), pm(1), pp(1);
        p0 << u0; pm << u0 - h; pp << u0 + h;

        auto R0 = eval_all_dense(nb, p0, 2);            // [R, R', R''] (1 × n)
        auto R_minus = eval_all_dense(nb, pm, 0)[0];   // (1 × n)
        auto R_plus  = eval_all_dense(nb, pp, 0)[0];   // (1 × n)
        auto R_pm2   = eval_all_dense(nb, p0, 1);      // for 2nd-order FD on R'

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

/**
 * NURBS knot insertion preserves the curve geometry exactly (including
 * the rational weighting). The transform is built so it applies directly
 * to unweighted control points; the new weights are baked into the basis.
 */
TEST_CASE("NURBS: insert_knot preserves geometry", "[basis][nurbs]") {
    // Quarter-circle NURBS: degree 2, knot vector [0,0,0, 1,1,1], weights [1, sqrt(2)/2, 1]
    const Index p = 2;
    KnotVector<double> kv(std::vector<double>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
    Vector<double> w(3);
    w << 1.0, std::sqrt(2.0) / 2.0, 1.0;
    NURBS<double> nb(p, kv, w);

    // Control points for a unit quarter circle in the xy-plane.
    Eigen::MatrixX3d cps(3, 3);
    cps << 1.0, 0.0, 0.0,
           1.0, 1.0, 0.0,
           0.0, 1.0, 0.0;

    Eigen::VectorXd samples(6);
    samples << 0.05, 0.2, 0.4, 0.55, 0.78, 0.95;

    auto [basis, transform] = nb.insert_knot(0.5);
    REQUIRE(transform.rows() == 4);
    REQUIRE(transform.cols() == 3);

    Eigen::MatrixX3d new_cps = transform * cps;
    auto R_old = eval_all_dense(nb, samples, 0)[0];
    auto R_new = eval_all_dense(*basis, samples, 0)[0];
    Eigen::MatrixX3d c_old = R_old * cps;
    Eigen::MatrixX3d c_new = R_new * new_cps;
    REQUIRE((c_old - c_new).cwiseAbs().maxCoeff() < 1e-12);

    // The refined curve is still a unit circle (radius = 1) at each sample.
    for (Index i = 0; i < c_new.rows(); ++i) {
        const double r = c_new.row(i).head<2>().norm();
        REQUIRE(r == Approx(1.0).margin(1e-12));
    }
}
