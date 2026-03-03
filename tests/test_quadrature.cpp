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
        GaussLegendre<double> gl(0);
        REQUIRE(gl.num_points() == 0);
        REQUIRE(gl.points().size() == 0);
        REQUIRE(gl.weights().size() == 0);
    }

    SECTION("One point (precomputed)") {
        GaussLegendre<double> gl(1);
        REQUIRE(gl.num_points() == 1);
        REQUIRE(gl.points()(0) == Approx(0.0));
        REQUIRE(gl.weights()(0) == Approx(2.0));
    }

    SECTION("Two points (precomputed)") {
        GaussLegendre<double> gl(2);
        REQUIRE(gl.num_points() == 2);
        REQUIRE(gl.points()(0) == Approx(-0.577350269));
        REQUIRE(gl.points()(1) == Approx(0.577350269));
        REQUIRE(gl.weights()(0) == Approx(1.0));
        REQUIRE(gl.weights()(1) == Approx(1.0));
    }

    SECTION("Eight points (precomputed)") {
        GaussLegendre<double> gl(8);
        REQUIRE(gl.num_points() == 8);
        // Just checking bounds
        REQUIRE(gl.points()(0) < 0);
        REQUIRE(gl.points()(7) > 0);
        REQUIRE(gl.weights().sum() == Approx(2.0));
    }
}

/**
 * Gauss-Legendre computation test (no precomputed values).
 */
TEST_CASE("GaussLegendre computation", "[quadrature][gauss_legendre]") {
    SECTION("Computation (10 points)") {
        GaussLegendre<double> gl(10);
        REQUIRE(gl.num_points() == 10);
        
        // Sum of weights should be exactly 2 for domain [-1, 1]
        REQUIRE(gl.weights().sum() == Approx(2.0));
        
        // Integrate f(x) = 1, x, x^2
        double sum_1 = 0.0;
        double sum_x = 0.0;
        double sum_xx = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double w = gl.weights()(i);
            double x = gl.points()(i);
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
 * 1D domain mapping test.
 */
TEST_CASE("QuadratureRule 1D map_to_domain", "[quadrature]") {
    GaussLegendre<double> gl(3);
    auto [pts, wts] = gl.map_to_domain(0.0, 2.0);
    // Sum of mapped weights should equal the interval length
    REQUIRE(wts.sum() == Approx(2.0));
    // Points should be in [0, 2]
    for (std::size_t i = 0; i < gl.num_points(); ++i) {
        REQUIRE(pts(i) >= 0.0);
        REQUIRE(pts(i) <= 2.0);
    }
}

/**
 * Tensor product quadrature rule test.
 */
TEST_CASE("QuadratureRule tensor product", "[quadrature][gauss_legendre]") {
    GaussLegendre<double> gl2(2);
    GaussLegendre<double> gl3(3);

    SECTION("Isotropic 2D tensor product") {
        std::array<const QuadratureRule<double>*, 2> rules = {&gl3, &gl3};
        auto [pts, wts] = tensor_product<double, 2>(rules);
        REQUIRE(pts.rows() == 9);
        REQUIRE(pts.cols() == 2);
        REQUIRE(wts.sum() == Approx(4.0));
    }

    SECTION("Anisotropic 2D tensor product") {
        std::array<const QuadratureRule<double>*, 2> rules = {&gl2, &gl3};
        auto [pts, wts] = tensor_product<double, 2>(rules);
        REQUIRE(pts.rows() == 6);
        REQUIRE(pts.cols() == 2);
        REQUIRE(wts.sum() == Approx(4.0));
    }

    SECTION("Isotropic 3D tensor product") {
        std::array<const QuadratureRule<double>*, 3> rules = {&gl2, &gl2, &gl2};
        auto [pts, wts] = tensor_product<double, 3>(rules);
        REQUIRE(pts.rows() == 8);
        REQUIRE(pts.cols() == 3);
        REQUIRE(wts.sum() == Approx(8.0));
    }
}

/**
 * Gauss-Legendre 1D integration test.
 */
TEST_CASE("GaussLegendre 1D integration", "[quadrature][gauss_legendre]") {
    SECTION("Polynomial up to degree 5") {
        GaussLegendre<double> gl(3);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double x = gl.points()(i);
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
        GaussLegendre<double> gl(10);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gl.num_points(); ++i) {
            double x = gl.points()(i);
            integral += gl.weights()(i) * std::sin(x);
        }
        
        // Analytical integral of sin(x) from -1 to 1 is exactly 0 
        // as it is an odd function over a symmetric interval.
        REQUIRE(integral == Approx(0.0).margin(1e-12));
    }
}

/**
 * Gauss-Legendre 2D integration test (via tensor product).
 */
TEST_CASE("GaussLegendre 2D integration", "[quadrature][gauss_legendre]") {
    SECTION("Polynomial x^2 * y^2") {
        GaussLegendre<double> gl(3);
        std::array<const QuadratureRule<double>*, 2> rules = {&gl, &gl};
        auto [pts, wts] = tensor_product<double, 2>(rules);
        
        double integral = 0.0;
        for (Eigen::Index i = 0; i < pts.rows(); ++i) {
            double x = pts(i, 0);
            double y = pts(i, 1);
            integral += wts(i) * (x * x * y * y);
        }
        
        // Analytical integral from [-1, 1] x [-1, 1]:
        // (int_{-1}^1 x^2 dx) * (int_{-1}^1 y^2 dy) = (2/3) * (2/3) = 4/9
        REQUIRE(integral == Approx(4.0 / 9.0).margin(1e-12));
    }
}

/**
 * Gauss-Legendre 3D integration test (via tensor product).
 */
TEST_CASE("GaussLegendre 3D integration", "[quadrature][gauss_legendre]") {
    SECTION("Polynomial x^2 + y^2 + z^2") {
        GaussLegendre<double> gl(3);
        std::array<const QuadratureRule<double>*, 3> rules = {&gl, &gl, &gl};
        auto [pts, wts] = tensor_product<double, 3>(rules);
        
        double integral = 0.0;
        for (Eigen::Index i = 0; i < pts.rows(); ++i) {
            double x = pts(i, 0);
            double y = pts(i, 1);
            double z = pts(i, 2);
            integral += wts(i) * (x * x + y * y + z * z);
        }
        
        // Analytical integral from [-1, 1] x [-1, 1] x [-1, 1]:
        // 3 * (int_{-1}^1 x^2 dx) * (int_{-1}^1 1 dy) * (int_{-1}^1 1 dz)
        // = 3 * (2/3) * 2 * 2 = 8
        REQUIRE(integral == Approx(8.0).margin(1e-12));
    }
}
