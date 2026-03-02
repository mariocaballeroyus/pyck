#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bspline.hpp"
#include "knots.hpp"
#include "tensor.hpp"

using namespace pyck;

/**
 * Helper: build a (M*N x 2) parameter grid from 1D vectors u and v.
 * Row ordering is (u_0,v_0), (u_0,v_1), ..., (u_0,v_{N-1}),
 *                  (u_1,v_0), ..., i.e. v varies fastest...
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
 * Multivariate partition of unity.
 */
TEST_CASE("TensorProduct: partition of unity", "[basis][tensor]") {
    auto kv_u = clamped_uniform_knots<double>(2, 4);
    auto kv_v = clamped_uniform_knots<double>(3, 5);
    auto bu = std::make_shared<BSpline<double>>(2, kv_u);
    auto bv = std::make_shared<BSpline<double>>(3, kv_v);
    TensorProduct<double, 2> tp(std::array<Ptr<const Basis<double>>, 2>{bu, bv});

    auto u_pts = Eigen::VectorXd::LinSpaced(6, 0.0, 1.0);
    auto v_pts = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);

    for (int p = 0; p < u_pts.size(); ++p) {
        for (int q = 0; q < v_pts.size(); ++q) {
            Eigen::MatrixXd params(1, 2);
            params(0, 0) = u_pts(p);
            params(0, 1) = v_pts(q);
            std::array<Index, 2> spans = {
                bu->find_span(u_pts(p)),
                bv->find_span(v_pts(q))
            };
            Eigen::MatrixXd R = tp.eval(params, spans);
            REQUIRE(R.row(0).sum() == Approx(1.0).margin(1e-12));
        }
    }
}

/**
 * Lexicographical ordering of eval_derivs.
 */
TEST_CASE("TensorProduct: lexicographical ordering", "[basis][tensor]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    KnotVector<double> kv(knots);
    auto bu = std::make_shared<BSpline<double>>(2, kv);
    auto bv = std::make_shared<BSpline<double>>(2, kv);
    TensorProduct<double, 2> tp(std::array<Ptr<const Basis<double>>, 2>{bu, bv});

    Eigen::MatrixXd params(1, 2);
    params << 0.5, 0.5;
    std::array<Index, 2> spans = {2, 2};

    Eigen::VectorXd u(1);
    u << 0.5;
    Index span_1d = 2;
    auto du = bu->eval_derivs(u, span_1d, 1);  // [N_u, dN_u]
    auto dv = bv->eval_derivs(u, span_1d, 1);  // [N_v, dN_v]

    auto derivs = tp.eval_derivs(params, spans, Index(1));
    REQUIRE(derivs.size() == 4);

    std::size_t n_u = du[0].cols();  // p+1 = 3
    std::size_t n_v = dv[0].cols();  // p+1 = 3

    // Index 0 = (0,0): N_u ⊗ N_v
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(derivs[0](0, a * n_v + b) ==
                    Approx(du[0](0, a) * dv[0](0, b)).margin(1e-14));

    // Index 1 = (0,1): N_u ⊗ dN_v
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(derivs[1](0, a * n_v + b) ==
                    Approx(du[0](0, a) * dv[1](0, b)).margin(1e-14));

    // Index 2 = (1,0): dN_u ⊗ N_v
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(derivs[2](0, a * n_v + b) ==
                    Approx(du[1](0, a) * dv[0](0, b)).margin(1e-14));

    // Index 3 = (1,1): dN_u ⊗ dN_v
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(derivs[3](0, a * n_v + b) ==
                    Approx(du[1](0, a) * dv[1](0, b)).margin(1e-14));
}

/**
 * Linear consistency (bilinear patch).
 */
TEST_CASE("TensorProduct: bilinear consistency", "[basis][tensor]") {
    KnotVector<double> kv({0, 0, 1, 1});
    auto bu = std::make_shared<BSpline<double>>(1, kv);
    auto bv = std::make_shared<BSpline<double>>(1, kv);
    TensorProduct<double, 2> tp(std::array<Ptr<const Basis<double>>, 2>{bu, bv});

    // f(u,v) = 3u + 2v + 5uv + 1
    // Control values at corners (u-major, v-minor ordering):
    //   (0,0) → 1,  (0,1) → 3,  (1,0) → 4,  (1,1) → 11
    Eigen::VectorXd ctrl(4);
    ctrl << 1.0, 3.0, 4.0, 11.0;

    // Single span → spans = {1, 1}
    std::array<Index, 2> spans = {1, 1};
    auto params = make_grid(Eigen::VectorXd::LinSpaced(7, 0.0, 1.0),
                            Eigen::VectorXd::LinSpaced(7, 0.0, 1.0));
    Eigen::MatrixXd R = tp.eval(params, spans);  // (49 x 4)

    // Compute f(u,v) = R * ctrl and compare to analytical
    Eigen::VectorXd f_computed = R * ctrl;

    for (int i = 0; i < params.rows(); ++i) {
        double uu = params(i, 0), vv = params(i, 1);
        double f_exact = 3 * uu + 2 * vv + 5 * uu * vv + 1;
        REQUIRE(f_computed(i) == Approx(f_exact).margin(1e-12));
    }
}

/**
 * Derivative Kronecker product check.
 */
TEST_CASE("TensorProduct: derivative Kronecker check", "[basis][tensor]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    KnotVector<double> kv(knots);
    auto bu = std::make_shared<BSpline<double>>(2, kv);
    auto bv = std::make_shared<BSpline<double>>(2, kv);
    TensorProduct<double, 2> tp(std::array<Ptr<const Basis<double>>, 2>{bu, bv});

    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.7;
    std::array<Index, 2> spans = {2, 2};

    Eigen::VectorXd uu(1), vv(1);
    uu << 0.4;
    vv << 0.7;

    Index span_1d = 2;
    auto du = bu->eval_derivs(uu, span_1d, 1);
    auto dv = bv->eval_derivs(vv, span_1d, 1);
    
    auto derivs = tp.eval_derivs(params, spans, Index(1));

    std::size_t n_u = du[0].cols();  // p+1 = 3
    std::size_t n_v = dv[0].cols();  // p+1 = 3

    // (1,0): dN_u ⊗ N_v
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(derivs[2](0, a * n_v + b) ==
                    Approx(du[1](0, a) * dv[0](0, b)).margin(1e-14));

    // (0,1): N_u ⊗ dN_v
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(derivs[1](0, a * n_v + b) ==
                    Approx(du[0](0, a) * dv[1](0, b)).margin(1e-14));

    // (1,1): dN_u ⊗ dN_v
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(derivs[3](0, a * n_v + b) ==
                    Approx(du[1](0, a) * dv[1](0, b)).margin(1e-14));
}

/**
 * Finite-difference derivative check.
 */
TEST_CASE("TensorProduct: finite-difference derivative check", "[basis][tensor]") {
    auto kv_u = clamped_uniform_knots<double>(3, 5);
    auto kv_v = clamped_uniform_knots<double>(2, 4);
    auto bu = std::make_shared<BSpline<double>>(3, kv_u);
    auto bv = std::make_shared<BSpline<double>>(2, kv_v);
    TensorProduct<double, 2> tp(std::array<Ptr<const Basis<double>>, 2>{bu, bv});

    double h = 1e-7;
    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.6;
    std::array<Index, 2> spans = {
        bu->find_span(0.4),
        bv->find_span(0.6)
    };

    auto derivs = tp.eval_derivs(params, spans, Index(1));

    // ∂/∂u  (use same spans — for small h they don't change)
    Eigen::MatrixXd p_fwd(1, 2), p_bwd(1, 2);
    p_fwd << 0.4 + h, 0.6;
    p_bwd << 0.4 - h, 0.6;
    Eigen::MatrixXd dRdu_fd = (tp.eval(p_fwd, spans) - tp.eval(p_bwd, spans)) / (2.0 * h);

    for (int j = 0; j < dRdu_fd.cols(); ++j)
        REQUIRE(derivs[2](0, j) == Approx(dRdu_fd(0, j)).margin(1e-5));

    // ∂/∂v
    p_fwd << 0.4, 0.6 + h;
    p_bwd << 0.4, 0.6 - h;
    Eigen::MatrixXd dRdv_fd = (tp.eval(p_fwd, spans) - tp.eval(p_bwd, spans)) / (2.0 * h);

    for (int j = 0; j < dRdv_fd.cols(); ++j)
        REQUIRE(derivs[1](0, j) == Approx(dRdv_fd(0, j)).margin(1e-5));
}
