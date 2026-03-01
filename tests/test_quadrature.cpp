#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <cmath>

#include "gauss_legendre.hpp"

using namespace pyck;

/**
 * Gauss-Legendre functionality test.
 */
TEST_CASE("GaussLegendre functionality", "[quadrature][gauss_legendre]") {
    SECTION("Zero points") {
        GaussLegendre<double, 1> gl(0);
        REQUIRE(gl.num_points() == 0);
        REQUIRE(gl.dim() == 1);
        REQUIRE(gl.points().rows() == 0);
        REQUIRE(gl.weights().size() == 0);
    }

    SECTION("One point (precomputed)") {
        GaussLegendre<double, 1> gl(1);
        REQUIRE(gl.num_points() == 1);
        REQUIRE(gl.dim() == 1);
        REQUIRE(gl.points()(0, 0) == Approx(0.0));
        REQUIRE(gl.weights()(0) == Approx(2.0));
    }

    SECTION("Two points (precomputed)") {
        GaussLegendre<double, 1> gl(2);
        REQUIRE(gl.num_points() == 2);
        REQUIRE(gl.dim() == 1);
        REQUIRE(gl.points()(0, 0) == Approx(-0.577350269));
        REQUIRE(gl.points()(1, 0) == Approx(0.577350269));
        REQUIRE(gl.weights()(0) == Approx(1.0));
        REQUIRE(gl.weights()(1) == Approx(1.0));
    }

    SECTION("Eight points (precomputed)") {
        GaussLegendre<double, 1> gl(8);
        REQUIRE(gl.num_points() == 8);
        REQUIRE(gl.dim() == 1);
        // Just checking bounds
        REQUIRE(gl.points()(0, 0) < 0);
        REQUIRE(gl.points()(7, 0) > 0);
        REQUIRE(gl.weights().sum() == Approx(2.0));
    }
}

/**
 * Gauss-Legendre computation test (no precomputed values).
 */
TEST_CASE("GaussLegendre computation", "[quadrature][gauss_legendre]") {
    SECTION("Computation (10 points)") {
        GaussLegendre<double, 1> gl(10);
        REQUIRE(gl.num_points() == 10);
        REQUIRE(gl.dim() == 1);
        
        // Sum of weights should be exactly 2 for domain [-1, 1]
        REQUIRE(gl.weights().sum() == Approx(2.0));
        
        // Integrate f(x) = 1, x, x^2
        double sum_1 = 0.0;
        double sum_x = 0.0;
        double sum_xx = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double w = gl.weights()(i);
            double x = gl.points()(i, 0);
            sum_1 += w * 1.0;
            sum_x += w * x;
            sum_xx += w * x * x;
        }
        REQUIRE(sum_1 == Approx(2.0));
        REQUIRE(sum_x == Approx(0.0).margin(1e-10));
        REQUIRE(sum_xx == Approx(2.0 / 3.0));
    }
}

/**
 * Tensor product quadrature rule test.
 */
TEST_CASE("QuadratureRule tensor product", "[quadrature][gauss_legendre]") {
    std::array<std::size_t, 2> num_pts = {2, 3};
    GaussLegendre<double, 2> prod(num_pts);
    REQUIRE(prod.num_points() == 6);
    REQUIRE(prod.dim() == 2);
    REQUIRE(prod.weights().sum() == Approx(4.0));
}

/**
 * Gauss-Legendre 1D integration test.
 */
TEST_CASE("GaussLegendre 1D integration", "[quadrature][gauss_legendre]") {
    SECTION("Polynomial up to degree 5") {
        GaussLegendre<double, 1> gl(3);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double x = gl.points()(i, 0);
            // f(x) = 3x^5 - 2x^4 + x^2 - 7
            integral += gl.weights()(i) * (3 * std::pow(x, 5) - 2 * std::pow(x, 4) + x * x - 7);
        }
        
        // Analytical integral from -1 to 1:
        // [-2x^4] -> -4/5
        // [x^2] -> 2/3
        // [-7] -> -14
        // Result: -4/5 + 2/3 - 14 = -212/15
        REQUIRE(integral == Approx(-212.0 / 15.0).margin(1e-12));
    }
    
    SECTION("Trigonometric function (sine)") {
        GaussLegendre<double, 1> gl(10);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double x = gl.points()(i, 0);
            integral += gl.weights()(i) * std::sin(x);
        }
        
        // Analytical integral of sin(x) from -1 to 1 is exactly 0 
        // as it is an odd function over a symmetric interval.
        REQUIRE(integral == Approx(0.0).margin(1e-12));
    }
}

/**
 * Gauss-Legendre 2D integration test.
 */
TEST_CASE("GaussLegendre 2D integration", "[quadrature][gauss_legendre]") {
    SECTION("Polynomial x^2 * y^2") {
        GaussLegendre<double, 2> gl(3);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double x = gl.points()(i, 0);
            double y = gl.points()(i, 1);
            integral += gl.weights()(i) * (x * x * y * y);
        }
        
        // Analytical integral from [-1, 1] x [-1, 1]:
        // (int_{-1}^1 x^2 dx) * (int_{-1}^1 y^2 dy) = (2/3) * (2/3) = 4/9
        REQUIRE(integral == Approx(4.0 / 9.0).margin(1e-12));
    }
}

/**
 * Gauss-Legendre 3D integration test.
 */
TEST_CASE("GaussLegendre 3D integration", "[quadrature][gauss_legendre]") {
    SECTION("Polynomial x^2 + y^2 + z^2") {
        GaussLegendre<double, 3> gl(3);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double x = gl.points()(i, 0);
            double y = gl.points()(i, 1);
            double z = gl.points()(i, 2);
            integral += gl.weights()(i) * (x * x + y * y + z * z);
        }
        
        // Analytical integral from [-1, 1] x [-1, 1] x [-1, 1]:
        // 3 * (int_{-1}^1 x^2 dx) * (int_{-1}^1 1 dy) * (int_{-1}^1 1 dz)
        // = 3 * (2/3) * 2 * 2 = 8
        REQUIRE(integral == Approx(8.0).margin(1e-12));
    }
}
