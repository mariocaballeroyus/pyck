#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bspline.hpp"
#include "knots.hpp"
#include "tensor.hpp"

using namespace pyck;

/**
 * Helper: build a (M*N x 2) parameter grid from 1D vectors u and v.
 * Row ordering is (u_0,v_0), (u_0,v_1), ..., (u_0,v_{N-1}),
 *                  (u_1,v_0), ..., i.e. v varies fastest.
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
 *
 * For a 2D tensor-product basis R_{i,j}(u,v) = N_{i,p}(u) · M_{j,q}(v),
 * the sum over all basis functions must equal 1 at every point:
 *
 *   Σ_{i,j} R_{i,j}(u, v) = 1
 */
TEST_CASE("TensorProduct: partition of unity", "[basis][tensor]") {
    auto bu = std::make_shared<BSpline>(2, KnotVector::clamped_uniform(2, 4));
    auto bv = std::make_shared<BSpline>(3, KnotVector::clamped_uniform(3, 5));
    TensorProduct tp({bu, bv});

    auto params = make_grid(Eigen::VectorXd::LinSpaced(6, 0.0, 1.0),
                            Eigen::VectorXd::LinSpaced(5, 0.0, 1.0));
    Eigen::MatrixXd R = tp.eval(params);

    for (int i = 0; i < R.rows(); ++i)
        REQUIRE(R.row(i).sum() == Approx(1.0).margin(1e-12));
}

/**
 * Lexicographical ordering of eval_derivs.
 *
 * eval_derivs(params, {k_u, k_v}) returns a vector of matrices indexed
 * in lexicographic order of the derivative multi-index (i_u, i_v),
 * where i_v varies fastest:
 *
 *   index = i_u * (k_v + 1) + i_v
 *
 * For orders = {1, 1}, the 4 matrices correspond to:
 *   [0] = (0,0),  [1] = (0,1),  [2] = (1,0),  [3] = (1,1)
 *
 * We verify this against manually computed Kronecker products of 1D
 * basis values and derivatives (quadratic Bernstein at u = v = 0.5).
 */
TEST_CASE("TensorProduct: lexicographical ordering", "[basis][tensor]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    auto bu = std::make_shared<BSpline>(2, KnotVector(knots));
    auto bv = std::make_shared<BSpline>(2, KnotVector(knots));
    TensorProduct tp({bu, bv});

    Eigen::MatrixXd params(1, 2);
    params << 0.5, 0.5;

    Eigen::VectorXd u(1);
    u << 0.5;
    auto du = bu->eval_derivs(u, 1);  // [N_u, dN_u]
    auto dv = bv->eval_derivs(u, 1);  // [N_v, dN_v]

    auto derivs = tp.eval_derivs(params, {1, 1});
    REQUIRE(derivs.size() == 4);

    std::size_t n_u = bu->num_basis();
    std::size_t n_v = bv->num_basis();

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
 *
 * If both 1D bases are linear (degree 1, 2 basis functions each), the
 * tensor-product basis must be able to represent any bilinear function
 * exactly.  We test:
 *
 *   f(u, v) = 3u + 2v + 5uv + 1
 *
 * by computing  Σ R_{i,j}(u,v) · c_{i,j}  where the four control values
 * are f evaluated at the parameter corners (0,0), (0,1), (1,0), (1,1).
 */
TEST_CASE("TensorProduct: bilinear consistency", "[basis][tensor]") {
    auto bu = std::make_shared<BSpline>(1, KnotVector({0, 0, 1, 1}));
    auto bv = std::make_shared<BSpline>(1, KnotVector({0, 0, 1, 1}));
    TensorProduct tp({bu, bv});

    // f(u,v) = 3u + 2v + 5uv + 1
    // Control values at corners (u-major, v-minor ordering):
    //   (0,0) → 1,  (0,1) → 3,  (1,0) → 4,  (1,1) → 11
    Eigen::VectorXd ctrl(4);
    ctrl << 1.0, 3.0, 4.0, 11.0;

    auto params = make_grid(Eigen::VectorXd::LinSpaced(7, 0.0, 1.0),
                            Eigen::VectorXd::LinSpaced(7, 0.0, 1.0));
    Eigen::MatrixXd R = tp.eval(params);  // (49 x 4)

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
 *
 * For a 2D tensor-product basis, the mixed partial derivatives factor as:
 *
 *   ∂^{a+b} R / (∂u^a ∂v^b)  =  (d^a N_u / du^a) ⊗ (d^b N_v / dv^b)
 *
 * We verify (1,0), (0,1), and (1,1) against the manually computed
 * Kronecker products of the 1D quantities.
 */
TEST_CASE("TensorProduct: derivative Kronecker check", "[basis][tensor]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    auto bu = std::make_shared<BSpline>(2, KnotVector(knots));
    auto bv = std::make_shared<BSpline>(2, KnotVector(knots));
    TensorProduct tp({bu, bv});

    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.7;

    Eigen::VectorXd uu(1), vv(1);
    uu << 0.4;
    vv << 0.7;

    auto du = bu->eval_derivs(uu, 1);
    auto dv = bv->eval_derivs(vv, 1);
    auto derivs = tp.eval_derivs(params, {1, 1});

    std::size_t n_u = bu->num_basis();
    std::size_t n_v = bv->num_basis();

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
 *
 * We verify the parametric partial derivatives ∂R/∂u and ∂R/∂v returned
 * by eval_derivs against a central finite-difference approximation:
 *
 *   ∂R/∂u ≈ [R(u+h,v) - R(u-h,v)] / (2h)
 */
TEST_CASE("TensorProduct: finite-difference derivative check", "[basis][tensor]") {
    auto bu = std::make_shared<BSpline>(3, KnotVector::clamped_uniform(3, 5));
    auto bv = std::make_shared<BSpline>(2, KnotVector::clamped_uniform(2, 4));
    TensorProduct tp({bu, bv});

    double h = 1e-7;
    Eigen::MatrixXd params(1, 2);
    params << 0.4, 0.6;

    auto derivs = tp.eval_derivs(params, {1, 1});

    // ∂/∂u
    Eigen::MatrixXd p_fwd(1, 2), p_bwd(1, 2);
    p_fwd << 0.4 + h, 0.6;
    p_bwd << 0.4 - h, 0.6;
    Eigen::MatrixXd dRdu_fd = (tp.eval(p_fwd) - tp.eval(p_bwd)) / (2.0 * h);

    for (int j = 0; j < dRdu_fd.cols(); ++j)
        REQUIRE(derivs[2](0, j) == Approx(dRdu_fd(0, j)).margin(1e-5));

    // ∂/∂v
    p_fwd << 0.4, 0.6 + h;
    p_bwd << 0.4, 0.6 - h;
    Eigen::MatrixXd dRdv_fd = (tp.eval(p_fwd) - tp.eval(p_bwd)) / (2.0 * h);

    for (int j = 0; j < dRdv_fd.cols(); ++j)
        REQUIRE(derivs[1](0, j) == Approx(dRdv_fd(0, j)).margin(1e-5));
}
