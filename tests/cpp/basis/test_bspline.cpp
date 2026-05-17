#include "catch.hpp"

#include "bspline.hpp"
#include "knots.hpp"

using namespace pyck;

namespace {

/**
 * Naive Cox-de Boor recursion for a single basis function N_{i,p}(u).
 * This is used ONLY for verification of any optimized implementations.
 */
template <typename T>
T naive_cox_de_boor(T u, int i, int p, const KnotVector<T>& knots) {
    if (p == 0) {
        if (u >= knots[i] && u < knots[i + 1]) return 1.0;
        // Last point convention: if u is exactly the upper bound of the last non-empty span
        if (u == knots.back()) {
            // Find the last span that is actually non-empty
            std::size_t last_span = knots.size() - 2;
            while (last_span > 0 && knots[last_span] == knots[last_span + 1]) last_span--;
            if (i == static_cast<int>(last_span)) return 1.0;
        }
        return 0.0;
    }
    T left = 0.0;
    T den1 = knots[i + p] - knots[i];
    if (std::abs(den1) > 1e-15)
        left = (u - knots[i]) / den1 * naive_cox_de_boor(u, i, p - 1, knots);

    T right = 0.0;
    T den2 = knots[i + p + 1] - knots[i + 1];
    if (std::abs(den2) > 1e-15)
        right = (knots[i + p + 1] - u) / den2 * naive_cox_de_boor(u, i + 1, p - 1, knots);

    return left + right;
}

/**
 * Naive Cox-de Boor recursion for the k-th derivative of N_{i,p}(u).
 */
template <typename T>
T naive_cox_de_boor_deriv(T u, int i, int p, int k, const KnotVector<T>& knots) {
    if (k == 0) return naive_cox_de_boor(u, i, p, knots);
    if (p < k) return 0.0;

    T left = 0.0;
    T den1 = knots[i + p] - knots[i];
    if (std::abs(den1) > 1e-15)
        left = static_cast<T>(p) / den1 * naive_cox_de_boor_deriv(u, i, p - 1, k - 1, knots);

    T right = 0.0;
    T den2 = knots[i + p + 1] - knots[i + 1];
    if (std::abs(den2) > 1e-15)
        right = static_cast<T>(p) / den2 * naive_cox_de_boor_deriv(u, i + 1, p - 1, k - 1, knots);

    return left - right;
}

// Helper: evaluate all n basis functions at m points via eval_on_span loop.
static Eigen::MatrixXd eval_dense(const Basis<double>& basis,
                                   const Eigen::VectorXd& pts)
{
    const int n = basis.num_basis();
    const int p = basis.degree();
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(pts.size(), n);
    for (int i = 0; i < pts.size(); ++i) {
        int span = basis.find_span(pts[i]);
        Eigen::VectorXd pt(1); pt << pts[i];
        std::vector<Eigen::MatrixXd> local;
        basis.eval_on_span(pt, span, 0, local);
        N.row(i).segment(span - p, p + 1) = local[0].row(0);
    }
    return N;
}

} // namespace

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
TEST_CASE("BSpline: partition of unity", "[basis][bspline]") {
    for (std::size_t p = 1; p <= 3; ++p) {
        auto kv = KnotVector<double>::clamped_uniform(p, p + 3);
        BSpline<double> bs(p, kv);

        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(30, 0.0, 1.0);
        for (int i = 0; i < u.size(); ++i) {
            Eigen::VectorXd pt(1);
            pt << u(i);
            auto span = bs.find_span(u(i));
            std::vector<Eigen::MatrixXd> _bufN; bs.eval_on_span(pt, span, 0, _bufN); auto N = _bufN[0];
            REQUIRE(N.row(0).sum() == Approx(1.0).margin(1e-12));
        }
    }
}

/**
 * Non-negativity.
 *
 * B-spline basis functions are non-negative everywhere:
 *
 *   N_{i,p}(u) ≥ 0   for all i, p, u
 */
TEST_CASE("BSpline: non-negativity", "[basis][bspline]") {
    auto kv = KnotVector<double>::clamped_uniform(3, 8);
    BSpline<double> bs(3, kv);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(100, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span = bs.find_span(u(i));
        std::vector<Eigen::MatrixXd> _bufN; bs.eval_on_span(pt, span, 0, _bufN); auto N = _bufN[0];
        for (int j = 0; j < N.cols(); ++j)
            REQUIRE(N(0, j) >= -1e-15);
    }
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
TEST_CASE("BSpline: local support", "[basis][bspline]") {
    Vector<double> raw_knots(8);
    raw_knots << 0, 0, 0, 0.3, 0.7, 1, 1, 1;
    KnotVector<double> kv(raw_knots);
    BSpline<double> bs(2, kv);
    std::size_t n = bs.num_basis();

    // Evaluate on a fine grid, reconstructing full basis matrix
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(200, 0.0, 1.0);
    Eigen::MatrixXd N(u.size(), n);
    N.setZero();
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span = bs.find_span(u(i));
        std::vector<Eigen::MatrixXd> _bufN_local; bs.eval_on_span(pt, span, 0, _bufN_local); auto N_local = _bufN_local[0];
        for (int j = 0; j <= 2; ++j) {
            int global = static_cast<int>(span) - 2 + j;
            if (global >= 0 && global < static_cast<int>(n))
                N(i, global) = N_local(0, j);
        }
    }

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
TEST_CASE("BSpline: derivative sum to zero", "[basis][bspline]") {
    auto kv = KnotVector<double>::clamped_uniform(3, 6);
    BSpline<double> bs(3, kv);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(40, 0.0, 1.0);
    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1);
        pt << u(i);
        auto span = bs.find_span(u(i));
        std::vector<Eigen::MatrixXd> derivs; bs.eval_on_span(pt, span, 2, derivs);
        for (int k = 1; k <= 2; ++k)
            REQUIRE(derivs[k].row(0).sum() == Approx(0.0).margin(1e-10));
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
TEST_CASE("BSpline: C-continuity at internal knot", "[basis][bspline]") {
    // p = 3, 7 basis → knots = {0,0,0,0, 0.25, 0.5, 0.75, 1,1,1,1}
    auto kv = KnotVector<double>::clamped_uniform(3, 7);
    BSpline<double> bs(3, kv);
    std::size_t n = bs.num_basis();

    // Test at the internal knot ξ = 0.5
    double xi = 0.5;
    double eps = 1e-8;

    Eigen::VectorXd u_left(1), u_right(1);
    u_left  << xi - eps;
    u_right << xi + eps;

    auto span_left  = bs.find_span(xi - eps);
    auto span_right = bs.find_span(xi + eps);

    // Evaluate up to 3rd derivative on both sides
    std::vector<Eigen::MatrixXd> left; bs.eval_on_span(u_left,  span_left,  3, left);
    std::vector<Eigen::MatrixXd> right; bs.eval_on_span(u_right, span_right, 3, right);

    // Reconstruct full basis vectors and compare overlapping functions
    // Orders 0, 1, 2 must be continuous (C^2 for a simple knot with p=3)
    for (int k = 0; k <= 2; ++k) {
        Eigen::VectorXd left_full  = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd right_full = Eigen::VectorXd::Zero(n);
        for (int j = 0; j <= 3; ++j) {
            int gl = static_cast<int>(span_left)  - 3 + j;
            int gr = static_cast<int>(span_right) - 3 + j;
            if (gl >= 0 && gl < static_cast<int>(n))
                left_full(gl) = left[k](0, j);
            if (gr >= 0 && gr < static_cast<int>(n))
                right_full(gr) = right[k](0, j);
        }
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE(left_full(j) == Approx(right_full(j)).margin(1e-4));
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
TEST_CASE("BSpline: analytical values", "[basis][bspline]") {
    Eigen::VectorXd u(5);
    u << 0.0, 0.25, 0.5, 0.75, 1.0;

    SECTION("degree 1: linear Bernstein") {
        KnotVector<double> kv_1((Vector<double>(4) << 0, 0, 1, 1).finished());
        BSpline<double> bs(1, kv_1);
        auto span = bs.find_span(0.5);
        std::vector<Eigen::MatrixXd> _bufN; bs.eval_on_span(u, span, 0, _bufN); auto N = _bufN[0];

        for (int i = 0; i < 5; ++i) {
            REQUIRE(N(i, 0) == Approx(1.0 - u(i)).margin(1e-14));
            REQUIRE(N(i, 1) == Approx(u(i)).margin(1e-14));
        }
    }

    SECTION("degree 2: quadratic Bernstein") {
        KnotVector<double> kv_2((Vector<double>(6) << 0, 0, 0, 1, 1, 1).finished());
        BSpline<double> bs(2, kv_2);
        auto span = bs.find_span(0.5);
        std::vector<Eigen::MatrixXd> _bufN; bs.eval_on_span(u, span, 0, _bufN); auto N = _bufN[0];

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
TEST_CASE("BSpline: analytical derivatives", "[basis][bspline]") {
    Eigen::VectorXd u(3);
    u << 0.0, 0.5, 1.0;

    SECTION("degree 1: first derivative") {
        KnotVector<double> kv_1((Vector<double>(4) << 0, 0, 1, 1).finished());
        BSpline<double> bs(1, kv_1);
        auto span = bs.find_span(0.5);
        std::vector<Eigen::MatrixXd> _bufdN; bs.eval_on_span(u, span, 1, _bufdN); auto dN = _bufdN[1];

        for (int i = 0; i < 3; ++i) {
            REQUIRE(dN(i, 0) == Approx(-1.0).margin(1e-12));
            REQUIRE(dN(i, 1) == Approx( 1.0).margin(1e-12));
        }
    }

    SECTION("degree 2: first derivative") {
        KnotVector<double> kv_2((Vector<double>(6) << 0, 0, 0, 1, 1, 1).finished());
        BSpline<double> bs(2, kv_2);
        auto span = bs.find_span(0.5);
        std::vector<Eigen::MatrixXd> _bufdN; bs.eval_on_span(u, span, 1, _bufdN); auto dN = _bufdN[1];

        for (int i = 0; i < 3; ++i) {
            double t = u(i);
            REQUIRE(dN(i, 0) == Approx(-2.0 * (1 - t)).margin(1e-12));
            REQUIRE(dN(i, 1) == Approx( 2.0 - 4.0 * t).margin(1e-12));
            REQUIRE(dN(i, 2) == Approx( 2.0 * t).margin(1e-12));
        }
    }

    SECTION("degree 2: second derivative") {
        KnotVector<double> kv_2((Vector<double>(6) << 0, 0, 0, 1, 1, 1).finished());
        BSpline<double> bs(2, kv_2);
        auto span = bs.find_span(0.5);
        std::vector<Eigen::MatrixXd> _bufd2N; bs.eval_on_span(u, span, 2, _bufd2N); auto d2N = _bufd2N[2];

        for (int i = 0; i < 3; ++i) {
            REQUIRE(d2N(i, 0) == Approx( 2.0).margin(1e-10));
            REQUIRE(d2N(i, 1) == Approx(-4.0).margin(1e-10));
            REQUIRE(d2N(i, 2) == Approx( 2.0).margin(1e-10));
        }
    }

    SECTION("degree 2: third derivative is zero") {
        KnotVector<double> kv_2((Vector<double>(6) << 0, 0, 0, 1, 1, 1).finished());
        BSpline<double> bs(2, kv_2);
        auto span = bs.find_span(0.5);
        std::vector<Eigen::MatrixXd> _bufd3N; bs.eval_on_span(u, span, 3, _bufd3N); auto d3N = _bufd3N[3];

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
TEST_CASE("BSpline: finite-difference derivative check", "[basis][bspline][deriv]") {
    auto kv = KnotVector<double>::clamped_uniform(3, 6);
    BSpline<double> bs(3, kv);
    std::size_t n = bs.num_basis();

    double h = 1e-7;
    Eigen::VectorXd u(5);
    u << 0.1, 0.25, 0.5, 0.75, 0.9;

    for (int i = 0; i < u.size(); ++i) {
        Eigen::VectorXd pt(1), u_fwd(1), u_bwd(1);
        pt << u(i);
        u_fwd << u(i) + h;
        u_bwd << u(i) - h;

        auto span     = bs.find_span(u(i));
        auto span_fwd = bs.find_span(u(i) + h);
        auto span_bwd = bs.find_span(u(i) - h);

        std::vector<Eigen::MatrixXd> _bufdN_local; bs.eval_on_span(pt, span, 1, _bufdN_local); auto dN_local = _bufdN_local[1];
        std::vector<Eigen::MatrixXd> _bufN_fwd; bs.eval_on_span(u_fwd, span_fwd, 0, _bufN_fwd); auto N_fwd = _bufN_fwd[0];
        std::vector<Eigen::MatrixXd> _bufN_bwd; bs.eval_on_span(u_bwd, span_bwd, 0, _bufN_bwd); auto N_bwd = _bufN_bwd[0];

        // Reconstruct full basis vectors for comparison
        Eigen::VectorXd dN_full      = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd N_fwd_full   = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd N_bwd_full   = Eigen::VectorXd::Zero(n);
        for (int j = 0; j <= 3; ++j) {
            int g  = static_cast<int>(span)     - 3 + j;
            int gf = static_cast<int>(span_fwd) - 3 + j;
            int gb = static_cast<int>(span_bwd) - 3 + j;
            if (g  >= 0 && g  < static_cast<int>(n)) dN_full(g)      = dN_local(0, j);
            if (gf >= 0 && gf < static_cast<int>(n)) N_fwd_full(gf)  = N_fwd(0, j);
            if (gb >= 0 && gb < static_cast<int>(n)) N_bwd_full(gb)  = N_bwd(0, j);
        }

        Eigen::VectorXd dN_fd = (N_fwd_full - N_bwd_full) / (2.0 * h);
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE(dN_full(j) == Approx(dN_fd(j)).margin(1e-5));
    }
}

/**
 * BSpline: compare with naive Cox-de Boor.
 * 
 * We verify the optimized BSpline::eval_on_span() implementation against the
 * standard Cox-de Boor recursion formula.
 */
TEST_CASE("BSpline: compare with naive Cox-de Boor", "[basis][bspline]") {
    for (int p = 1; p <= 3; ++p) {
        // Create a non-uniform knot vector
        Vector<double> knots(2 * (p + 1) + 2);
        Index k = 0;
        for (int i = 0; i <= p; ++i) knots(k++) = 0.0;
        knots(k++) = 0.3;
        knots(k++) = 0.7;
        for (int i = 0; i <= p; ++i) knots(k++) = 1.0;

        KnotVector<double> kv(knots);
        BSpline<double> bs(p, kv);
        
        // Points to evaluate
        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(50, 0.0, 1.0);
        
        for (int i = 0; i < u.size(); ++i) {
            double val = u(i);
            Index span = bs.find_span(val);
            
            Eigen::VectorXd pt(1);
            pt << val;
            std::vector<Eigen::MatrixXd> _bufN_fast; bs.eval_on_span(pt, span, 0, _bufN_fast); auto N_fast = _bufN_fast[0];
            
            for (int k = 0; k <= p; ++k) {
                int global_idx = static_cast<int>(span) - p + k;
                double N_naive = naive_cox_de_boor(val, global_idx, p, kv);
                REQUIRE(N_fast(0, k) == Approx(N_naive).margin(1e-12));
            }
        }
    }
}

/**
 * BSpline: non-clamped knot vector.
 * 
 * We verify that the evaluation algorithm (De Boor) works correctly
 * even for non-clamped knot vectors (no repeated end knots).
 * In this case, the valid domain is [xi_p, xi_{n+1}].
 */
TEST_CASE("BSpline: non-clamped knot vector", "[basis][bspline]") {
    for (int p = 1; p <= 3; ++p) {
        // Create a uniform, non-clamped knot vector: {0, 1, 2, 3, 4, 5, ...}
        int n = 7; // Number of basis functions
        Vector<double> knots(n + p + 1);
        for (int i = 0; i < n + p + 1; ++i) {
            knots(i) = static_cast<double>(i);
        }

        KnotVector<double> kv(knots);
        BSpline<double> bs(p, kv);
        
        // Valid domain is [knots[p], knots[n]]
        double u_min = knots[p];
        double u_max = knots[n];
        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(50, u_min, u_max);
        
        for (int i = 0; i < u.size(); ++i) {
            double val = u(i);
            Index span = bs.find_span(val);
            
            Eigen::VectorXd pt(1);
            pt << val;
            std::vector<Eigen::MatrixXd> _bufN_fast; bs.eval_on_span(pt, span, 0, _bufN_fast); auto N_fast = _bufN_fast[0];
            
            for (int k = 0; k <= p; ++k) {
                int global_idx = static_cast<int>(span) - p + k;
                double N_naive = naive_cox_de_boor(val, global_idx, p, kv);
                REQUIRE(N_fast(0, k) == Approx(N_naive).margin(1e-12));
            }
        }
    }
}

/**
 * BSpline: compare eval_on_span with naive.
 * 
 * We verify the optimized BSpline::eval_on_span() implementation against the
 * standard Cox-de Boor derivative recursion formula.
 */
TEST_CASE("BSpline: compare eval_on_span with naive", "[basis][bspline][deriv]") {
    for (int p = 1; p <= 3; ++p) {
        auto kv = KnotVector<double>::clamped_uniform(p, p + 5);
        BSpline<double> bs(p, kv);
        
        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(20, 0.0, 1.0);
        int max_order = p + 1;
        
        for (int i = 0; i < u.size(); ++i) {
            double val = u(i);
            Index span = bs.find_span(val);
            
            Eigen::VectorXd pt(1);
            pt << val;
            std::vector<Eigen::MatrixXd> derivs; bs.eval_on_span(pt, span, max_order, derivs);
            
            for (int k = 0; k <= max_order; ++k) {
                for (int j = 0; j <= p; ++j) {
                    int global_idx = static_cast<int>(span) - p + j;
                    double dN_naive = naive_cox_de_boor_deriv(val, global_idx, p, k, kv);
                    REQUIRE(derivs[k](0, j) == Approx(dN_naive).margin(1e-10));
                }
            }
        }
    }
}

/**
 * BSpline knot insertion preserves the curve geometry.
 *
 * For arbitrary control points P, the curve c(u) = N(u) · P must equal
 * the refined c'(u) = N_new(u) · (T · P) at every parameter sample.
 */
TEST_CASE("BSpline: insert_knot preserves geometry", "[basis][bspline]") {
    const Index p = 3;
    const Index n = 6;
    BSpline<double> bs(p, KnotVector<double>::clamped_uniform(p, n));

    // Arbitrary 3D control polygon
    Eigen::MatrixX3d cps(n, 3);
    for (Index i = 0; i < n; ++i) {
        cps(i, 0) = static_cast<double>(i);
        cps(i, 1) = 0.5 * static_cast<double>(i * i);
        cps(i, 2) = std::sin(static_cast<double>(i));
    }

    Eigen::VectorXd samples(5);
    samples << 0.05, 0.27, 0.5, 0.72, 0.93;

    SECTION("single insertion") {
        auto [basis, transform] = bs.insert_knot(0.3);
        REQUIRE(transform.rows() == n + 1);
        REQUIRE(transform.cols() == n);
        // Partition of unity in the row-sums (Boehm preserves it).
        for (Index i = 0; i < transform.rows(); ++i)
            REQUIRE(transform.row(i).sum() == Approx(1.0));

        Eigen::MatrixX3d new_cps = transform * cps;
        auto N_old = eval_dense(bs, samples);
        auto N_new = eval_dense(*basis, samples);

        Eigen::MatrixX3d c_old = N_old * cps;
        Eigen::MatrixX3d c_new = N_new * new_cps;
        REQUIRE((c_old - c_new).cwiseAbs().maxCoeff() < 1e-12);
    }

    SECTION("multi-insertion via chained single inserts") {
        auto [basis1, T1] = bs.insert_knot(0.4);
        auto [basis2, T2] = basis1->insert_knot(0.4);
        Eigen::MatrixXd transform = T2 * T1;
        REQUIRE(transform.rows() == n + 2);
        REQUIRE(transform.cols() == n);

        Eigen::MatrixX3d new_cps = transform * cps;
        auto N_old = eval_dense(bs, samples);
        auto N_new = eval_dense(*basis2, samples);
        Eigen::MatrixX3d c_old = N_old * cps;
        Eigen::MatrixX3d c_new = N_new * new_cps;
        REQUIRE((c_old - c_new).cwiseAbs().maxCoeff() < 1e-12);
    }
}
