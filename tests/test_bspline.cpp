#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bspline.hpp"

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
 * Test the constructor and accessors of the BSpline class
 */
TEST_CASE("BSpline: construction and accessors", "[bspline]") {
    std::vector<double> knots = {0, 0, 0, 0.5, 1, 1, 1};
    BSpline bs(2, knots);

    REQUIRE(bs.degree() == 2);
    REQUIRE(bs.num_basis() == 4);  // 7 - 2 - 1 = 4
    REQUIRE(bs.knots().size() == 7);
    REQUIRE(bs.knots() == knots);
}

/**
 * Test that eval() returns matrices of the correct size for various orders
 */
TEST_CASE("BSpline: eval returns correct structure", "[bspline]") {
    auto knots = clamped_uniform_knots(3, 6);
    BSpline bs(3, knots);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(10, 0.0, 1.0);

    SECTION("order 0 → single matrix") {
        auto result = bs.eval(u);
        REQUIRE(result.rows() == 10);
        REQUIRE(result.cols() == static_cast<int>(bs.num_basis()));
    }

    SECTION("order 2 → 3 matrices via eval_derivs") {
        auto result = bs.eval_derivs(u, 2);
        REQUIRE(result.size() == 3);
        for (auto& mat : result) {
            REQUIRE(mat.rows() == 10);
            REQUIRE(mat.cols() == static_cast<int>(bs.num_basis()));
        }
    }
}


/**
 * Test the partition of unity property: sum of basis functions should be 1 at 
 * all parameter values
 */
TEST_CASE("BSpline: partition of unity", "[bspline]") {
    SECTION("degree 1, 4 basis functions") {
        auto knots = clamped_uniform_knots(1, 4);
        BSpline bs(1, knots);

        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(20, 0.0, 1.0);
        auto N = bs.eval(u);

        REQUIRE(N.rows() == 20);
        REQUIRE(N.cols() == 4);

        for (int i = 0; i < N.rows(); ++i)
            REQUIRE(N.row(i).sum() == Approx(1.0).margin(1e-12));
    }

    SECTION("degree 2, 5 basis functions") {
        auto knots = clamped_uniform_knots(2, 5);
        BSpline bs(2, knots);

        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(30, 0.0, 1.0);
        auto N = bs.eval(u);

        REQUIRE(N.rows() == 30);
        REQUIRE(N.cols() == 5);

        for (int i = 0; i < N.rows(); ++i)
            REQUIRE(N.row(i).sum() == Approx(1.0).margin(1e-12));
    }

    SECTION("degree 3, 6 basis functions") {
        auto knots = clamped_uniform_knots(3, 6);
        BSpline bs(3, knots);

        Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(50, 0.0, 1.0);
        auto N = bs.eval(u);

        REQUIRE(N.rows() == 50);
        REQUIRE(N.cols() == 6);

        for (int i = 0; i < N.rows(); ++i)
            REQUIRE(N.row(i).sum() == Approx(1.0).margin(1e-12));
    }
}

/**
 * Test the non-negativity property: all basis function values should be >= 0
 */
TEST_CASE("BSpline: non-negativity", "[bspline]") {
    auto knots = clamped_uniform_knots(3, 8);
    BSpline bs(3, knots);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(100, 0.0, 1.0);
    auto N = bs.eval(u);

    for (int i = 0; i < N.rows(); ++i)
        for (int j = 0; j < N.cols(); ++j)
            REQUIRE(N(i, j) >= -1e-15);
}

/**
 * Test that the first and last basis functions interpolate the endpoints 
 * (u=0 and u=1)
 */
TEST_CASE("BSpline: endpoint interpolation", "[bspline]") {
    auto knots = clamped_uniform_knots(3, 7);
    BSpline bs(3, knots);

    Eigen::VectorXd u_start(1), u_end(1);
    u_start << 0.0;
    u_end   << 1.0;

    auto N0 = bs.eval(u_start);
    auto N1 = bs.eval(u_end);

    REQUIRE(N0(0, 0) == Approx(1.0).margin(1e-12));
    REQUIRE(N1(0, bs.num_basis() - 1) == Approx(1.0).margin(1e-12));

    for (std::size_t j = 1; j < bs.num_basis(); ++j)
        REQUIRE(N0(0, j) == Approx(0.0).margin(1e-12));
    for (std::size_t j = 0; j < bs.num_basis() - 1; ++j)
        REQUIRE(N1(0, j) == Approx(0.0).margin(1e-12));
}

/**
 * Test degree-1 basis functions against their known analytical expressions
 */
TEST_CASE("BSpline: degree-1 known values", "[bspline]") {
    // N_0(u) = 1-u,  N_1(u) = u
    std::vector<double> knots = {0, 0, 1, 1};
    BSpline bs(1, knots);

    REQUIRE(bs.num_basis() == 2);

    Eigen::VectorXd u(5);
    u << 0.0, 0.25, 0.5, 0.75, 1.0;

    auto N = bs.eval(u);

    for (int i = 0; i < 5; ++i) {
        REQUIRE(N(i, 0) == Approx(1.0 - u(i)).margin(1e-14));
        REQUIRE(N(i, 1) == Approx(u(i)).margin(1e-14));
    }
}

/**
 * Test degree-2 basis functions against their known analytical expressions 
 * at u=0.5
 */
TEST_CASE("BSpline: degree-2 midpoint values", "[bspline]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bs(2, knots);

    REQUIRE(bs.num_basis() == 3);

    Eigen::VectorXd u(1);
    u << 0.5;

    auto N = bs.eval(u);

    // N_0(0.5) = (1-u)^2 = 0.25
    // N_1(0.5) = 2u(1-u) = 0.5
    // N_2(0.5) = u^2     = 0.25
    REQUIRE(N(0, 0) == Approx(0.25).margin(1e-14));
    REQUIRE(N(0, 1) == Approx(0.50).margin(1e-14));
    REQUIRE(N(0, 2) == Approx(0.25).margin(1e-14));
}

/**
 * Test that the first derivative of the basis functions sums to zero at all 
 * parameter values
 */
TEST_CASE("BSpline: first derivative sums to zero", "[bspline][deriv]") {
    auto knots = clamped_uniform_knots(3, 6);
    BSpline bs(3, knots);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(40, 0.0, 1.0);
    auto result = bs.eval_derivs(u, 1);
    auto& dN = result[1];

    REQUIRE(dN.rows() == 40);
    REQUIRE(dN.cols() == static_cast<int>(bs.num_basis()));

    for (int i = 0; i < dN.rows(); ++i)
        REQUIRE(dN.row(i).sum() == Approx(0.0).margin(1e-10));
}

/**
 * Test degree-1 first derivative against known analytical expressions
 */
TEST_CASE("BSpline: degree-1 first derivative", "[bspline][deriv]") {
    // N_0'(u) = -1,  N_1'(u) = +1
    std::vector<double> knots = {0, 0, 1, 1};
    BSpline bs(1, knots);

    Eigen::VectorXd u(3);
    u << 0.0, 0.5, 1.0;

    auto dN = bs.eval_derivs(u, 1)[1];

    for (int i = 0; i < 3; ++i) {
        REQUIRE(dN(i, 0) == Approx(-1.0).margin(1e-12));
        REQUIRE(dN(i, 1) == Approx( 1.0).margin(1e-12));
    }
}

/**
 * Test degree-2 first derivative against known analytical expressions
 */
TEST_CASE("BSpline: degree-2 first derivative", "[bspline][deriv]") {
    // N_0 = (1-u)^2  → N_0' = -2(1-u)
    // N_1 = 2u(1-u)  → N_1' = 2 - 4u
    // N_2 = u^2      → N_2' = 2u
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bs(2, knots);

    Eigen::VectorXd u(3);
    u << 0.0, 0.5, 1.0;

    auto dN = bs.eval_derivs(u, 1)[1];

    // u = 0: [-2, 2, 0]
    REQUIRE(dN(0, 0) == Approx(-2.0).margin(1e-12));
    REQUIRE(dN(0, 1) == Approx( 2.0).margin(1e-12));
    REQUIRE(dN(0, 2) == Approx( 0.0).margin(1e-12));

    // u = 0.5: [-1, 0, 1]
    REQUIRE(dN(1, 0) == Approx(-1.0).margin(1e-12));
    REQUIRE(dN(1, 1) == Approx( 0.0).margin(1e-12));
    REQUIRE(dN(1, 2) == Approx( 1.0).margin(1e-12));

    // u = 1: [0, -2, 2]
    REQUIRE(dN(2, 0) == Approx( 0.0).margin(1e-12));
    REQUIRE(dN(2, 1) == Approx(-2.0).margin(1e-12));
    REQUIRE(dN(2, 2) == Approx( 2.0).margin(1e-12));
}

/**
 * Test degree-2 second derivative against known analytical expressions
 */
TEST_CASE("BSpline: degree-2 second derivative", "[bspline][deriv]") {
    // N_0'' = 2, N_1'' = -4, N_2'' = 2
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bs(2, knots);

    Eigen::VectorXd u(3);
    u << 0.0, 0.5, 1.0;

    auto d2N = bs.eval_derivs(u, 2)[2];

    for (int i = 0; i < 3; ++i) {
        REQUIRE(d2N(i, 0) == Approx( 2.0).margin(1e-10));
        REQUIRE(d2N(i, 1) == Approx(-4.0).margin(1e-10));
        REQUIRE(d2N(i, 2) == Approx( 2.0).margin(1e-10));
    }
}

/**
 * Test that the third derivative of degree-2 basis functions is zero at all 
 * parameter values
 */
TEST_CASE("BSpline: derivative higher than degree is zero", "[bspline][deriv]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    BSpline bs(2, knots);

    Eigen::VectorXd u(3);
    u << 0.0, 0.5, 1.0;

    auto d3N = bs.eval_derivs(u, 3)[3];

    for (int i = 0; i < d3N.rows(); ++i)
        for (int j = 0; j < d3N.cols(); ++j)
            REQUIRE(d3N(i, j) == Approx(0.0).margin(1e-14));
}

/**
 * Test that the free function evaluate_bspline produces the same results as the 
 * BSpline class method eval()
 */
TEST_CASE("eval_derivs order 0 matches eval", "[bspline]") {
    auto knots = clamped_uniform_knots(2, 5);
    BSpline bs(2, knots);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(12, 0.0, 1.0);

    auto derivs = bs.eval_derivs(u, 0);
    auto values = bs.eval(u);

    REQUIRE(derivs.size() == 1);
    REQUIRE(derivs[0].isApprox(values, 1e-14));
}

/**
 * Test the first derivative computed by eval() against a finite difference 
 * approximation
 */
TEST_CASE("BSpline: finite difference derivative check", "[bspline][deriv]") {
    auto knots = clamped_uniform_knots(3, 6);
    BSpline bs(3, knots);

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

/**
 * Test that the basis functions defined by a non-uniform knot vector still
 * satisfy the partition of unity and non-negativity properties
 */
TEST_CASE("BSpline: non-uniform knot vector", "[bspline]") {
    std::vector<double> knots = {0, 0, 0, 0.3, 0.7, 1, 1, 1};
    BSpline bs(2, knots);

    REQUIRE(bs.num_basis() == 5);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(50, 0.0, 1.0);
    auto N = bs.eval(u);

    for (int i = 0; i < N.rows(); ++i) {
        REQUIRE(N.row(i).sum() == Approx(1.0).margin(1e-12));
        for (int j = 0; j < N.cols(); ++j)
            REQUIRE(N(i, j) >= -1e-15);
    }
}

/**
 * Test that eval() can be called with a single parameter value and returns the
 * correct basis function values at that point
 */
TEST_CASE("BSpline: single evaluation point", "[bspline]") {
    auto knots = clamped_uniform_knots(2, 4);
    BSpline bs(2, knots);

    Eigen::VectorXd u(1);
    u << 0.5;

    auto N = bs.eval(u);
    REQUIRE(N.rows() == 1);
    REQUIRE(N.cols() == 4);
    REQUIRE(N.row(0).sum() == Approx(1.0).margin(1e-12));
}
