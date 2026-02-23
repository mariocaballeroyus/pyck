#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bspline.hpp"
#include "tensor.hpp"

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
 * Test the constructor and accessors of the TensorProduct class
 */
TEST_CASE("TensorProduct: construction and accessors", "[tensor]") {
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);

    auto bu = std::make_shared<BSpline>(2, ku);
    auto bv = std::make_shared<BSpline>(2, kv);

    SECTION("2D tensor product") {
        TensorProduct tp({bu, bv});
        REQUIRE(tp.dim() == 2);
        REQUIRE(tp.num_basis() == 4 * 3);
    }

    SECTION("1D (single basis)") {
        TensorProduct tp({bu});
        REQUIRE(tp.dim() == 1);
        REQUIRE(tp.num_basis() == 4);
    }

    SECTION("3D tensor product") {
        auto bw = std::make_shared<BSpline>(1, clamped_uniform_knots(1, 2));
        TensorProduct tp({bu, bv, bw});
        REQUIRE(tp.dim() == 3);
        REQUIRE(tp.num_basis() == 4 * 3 * 2);
    }
}

/**
 * Test that a TensorProduct with a single basis matches BSpline::eval exactly
 */
TEST_CASE("TensorProduct: 1D passthrough", "[tensor]") {
    auto knots = clamped_uniform_knots(2, 5);
    auto bs = std::make_shared<BSpline>(2, knots);
    TensorProduct tp({bs});

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(10, 0.0, 1.0);
    Eigen::MatrixXd params(10, 1);
    params.col(0) = u;

    Eigen::MatrixXd tp_result = tp.eval(params);
    Eigen::MatrixXd bs_result = bs->eval(u);

    REQUIRE(tp_result.rows() == bs_result.rows());
    REQUIRE(tp_result.cols() == bs_result.cols());
    REQUIRE(tp_result.isApprox(bs_result, 1e-14));
}

/**
 * Test partition of unity: row sums of eval() should be 1 at all parameter values
 */
TEST_CASE("TensorProduct: 2D partition of unity", "[tensor]") {
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(3, 5);
    auto bu = std::make_shared<BSpline>(2, ku);
    auto bv = std::make_shared<BSpline>(3, kv);
    TensorProduct tp({bu, bv});

    // Build a grid of evaluation points
    std::size_t M = 6, N = 5;
    Eigen::MatrixXd params(M * N, 2);
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(M, 0.0, 1.0);
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);
    for (std::size_t p = 0; p < M; ++p)
        for (std::size_t q = 0; q < N; ++q) {
            params(p * N + q, 0) = u(p);
            params(p * N + q, 1) = v(q);
        }

    Eigen::MatrixXd R = tp.eval(params);

    REQUIRE(R.rows() == static_cast<int>(M * N));
    REQUIRE(R.cols() == static_cast<int>(bu->num_basis() * bv->num_basis()));

    for (int i = 0; i < R.rows(); ++i)
        REQUIRE(R.row(i).sum() == Approx(1.0).margin(1e-12));
}

/**
 * Test 2D known values: compare eval() against manually-computed Kronecker product
 */
TEST_CASE("TensorProduct: 2D known values", "[tensor]") {
    // Quadratic Bernstein: knots = {0,0,0,1,1,1}
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    auto bu = std::make_shared<BSpline>(2, knots);
    auto bv = std::make_shared<BSpline>(2, knots);
    TensorProduct tp({bu, bv});

    // Evaluate at a single point (0.5, 0.5)
    Eigen::MatrixXd params(1, 2);
    params << 0.5, 0.5;

    Eigen::MatrixXd R = tp.eval(params);

    // 1D Bernstein at u=0.5: N = [0.25, 0.5, 0.25]
    Eigen::VectorXd u(1);
    u << 0.5;
    Eigen::MatrixXd Nu = bu->eval(u);
    Eigen::MatrixXd Nv = bv->eval(u);

    // Manual Kronecker product
    REQUIRE(R.cols() == 9);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            REQUIRE(R(0, a * 3 + b) == Approx(Nu(0, a) * Nv(0, b)).margin(1e-14));
}

/**
 * Test that eval_derivs returns the correct number and shape of matrices
 */
TEST_CASE("TensorProduct: derivatives structure", "[tensor][deriv]") {
    auto ku = clamped_uniform_knots(2, 4);
    auto kv = clamped_uniform_knots(2, 3);
    auto bu = std::make_shared<BSpline>(2, ku);
    auto bv = std::make_shared<BSpline>(2, kv);
    TensorProduct tp({bu, bv});

    Eigen::MatrixXd params(1, 2);
    params << 0.5, 0.5;

    SECTION("orders={1,1} → 4 matrices") {
        auto derivs = tp.eval_derivs(params, {1, 1});
        REQUIRE(derivs.size() == 4);  // (0,0), (0,1), (1,0), (1,1)
        for (auto& mat : derivs) {
            REQUIRE(mat.rows() == 1);
            REQUIRE(mat.cols() == static_cast<int>(tp.num_basis()));
        }
    }

    SECTION("orders={2,0} → 3 matrices") {
        auto derivs = tp.eval_derivs(params, {2, 0});
        REQUIRE(derivs.size() == 3);  // (0,0), (1,0), (2,0)
    }

    SECTION("orders={0,0} → 1 matrix matching eval") {
        auto derivs = tp.eval_derivs(params, {0, 0});
        REQUIRE(derivs.size() == 1);
        REQUIRE(derivs[0].isApprox(tp.eval(params), 1e-14));
    }
}

/**
 * Test derivative values against manual product of 1D derivative matrices
 */
TEST_CASE("TensorProduct: derivative values", "[tensor][deriv]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    auto bu = std::make_shared<BSpline>(2, knots);
    auto bv = std::make_shared<BSpline>(2, knots);
    TensorProduct tp({bu, bv});

    Eigen::MatrixXd params(1, 2);
    params << 0.5, 0.5;

    Eigen::VectorXd u(1), v(1);
    u << 0.5;
    v << 0.5;

    // Get 1D derivatives
    auto du = bu->eval_derivs(u, 1);  // [N, dN]
    auto dv = bv->eval_derivs(v, 1);

    auto derivs = tp.eval_derivs(params, {1, 1});
    // derivs ordering: (0,0), (0,1), (1,0), (1,1)

    std::size_t n_u = bu->num_basis();
    std::size_t n_v = bv->num_basis();

    // Check (1,0): dN_u × N_v
    auto& d10 = derivs[2]; // index = 1*(1+1) + 0
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(d10(0, a * n_v + b) ==
                    Approx(du[1](0, a) * dv[0](0, b)).margin(1e-14));

    // Check (0,1): N_u × dN_v
    auto& d01 = derivs[1]; // index = 0*(1+1) + 1
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(d01(0, a * n_v + b) ==
                    Approx(du[0](0, a) * dv[1](0, b)).margin(1e-14));

    // Check (1,1): dN_u × dN_v
    auto& d11 = derivs[3]; // index = 1*(1+1) + 1
    for (std::size_t a = 0; a < n_u; ++a)
        for (std::size_t b = 0; b < n_v; ++b)
            REQUIRE(d11(0, a * n_v + b) ==
                    Approx(du[1](0, a) * dv[1](0, b)).margin(1e-14));
}

/**
 * Test the non-negativity property: all basis function values should be >= 0
 */
TEST_CASE("TensorProduct: non-negativity", "[tensor]") {
    auto ku = clamped_uniform_knots(3, 6);
    auto kv = clamped_uniform_knots(2, 4);
    auto bu = std::make_shared<BSpline>(3, ku);
    auto bv = std::make_shared<BSpline>(2, kv);
    TensorProduct tp({bu, bv});

    std::size_t M = 10, N = 8;
    Eigen::MatrixXd params(M * N, 2);
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(M, 0.0, 1.0);
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);
    for (std::size_t p = 0; p < M; ++p)
        for (std::size_t q = 0; q < N; ++q) {
            params(p * N + q, 0) = u(p);
            params(p * N + q, 1) = v(q);
        }

    Eigen::MatrixXd R = tp.eval(params);

    for (int i = 0; i < R.rows(); ++i)
        for (int j = 0; j < R.cols(); ++j)
            REQUIRE(R(i, j) >= -1e-15);
}

/**
 * Verify first derivatives against central finite differences
 */
TEST_CASE("TensorProduct: finite difference derivative check", "[tensor][deriv]") {
    auto ku = clamped_uniform_knots(3, 5);
    auto kv = clamped_uniform_knots(2, 4);
    auto bu = std::make_shared<BSpline>(3, ku);
    auto bv = std::make_shared<BSpline>(2, kv);
    TensorProduct tp({bu, bv});

    double h = 1e-7;
    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.6;

    auto derivs = tp.eval_derivs(params, {1, 1});
    // derivs[2] = d/du at (0.4, 0.6), derivs[1] = d/dv

    // Finite difference for d/du
    Eigen::MatrixXd params_fwd(1, 2), params_bwd(1, 2);
    params_fwd << 0.4 + h, 0.6;
    params_bwd << 0.4 - h, 0.6;
    Eigen::MatrixXd dRdu_fd = (tp.eval(params_fwd) - tp.eval(params_bwd)) / (2.0 * h);

    for (int j = 0; j < dRdu_fd.cols(); ++j)
        REQUIRE(derivs[2](0, j) == Approx(dRdu_fd(0, j)).margin(1e-5));

    // Finite difference for d/dv
    params_fwd << 0.4, 0.6 + h;
    params_bwd << 0.4, 0.6 - h;
    Eigen::MatrixXd dRdv_fd = (tp.eval(params_fwd) - tp.eval(params_bwd)) / (2.0 * h);

    for (int j = 0; j < dRdv_fd.cols(); ++j)
        REQUIRE(derivs[1](0, j) == Approx(dRdv_fd(0, j)).margin(1e-5));
}
