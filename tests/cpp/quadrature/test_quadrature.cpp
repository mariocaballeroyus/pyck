#include "catch.hpp"
#include <cmath>

#include "gauss_legendre.hpp"

using namespace pyck;

/**
 * Gauss-Legendre computation test (no precomputed values).
 */
TEST_CASE("GaussLegendre computation", "[quadrature][gauss_legendre]") {
    SECTION("Computation (10 points)") {
        GaussLegendre<double, 1> gauss_rule(10);
        REQUIRE(gauss_rule.num_points() == 10);
        
        // Sum of weights should be exactly 2 for domain [-1, 1]
        REQUIRE(gauss_rule.weights().sum() == Approx(2.0));
        
        // Integrate f(x) = 1, x, x^2
        double sum_1 = 0.0;
        double sum_x = 0.0;
        double sum_xx = 0.0;
        for (std::size_t i = 0; i < gauss_rule.num_points(); ++i) {
            double w = gauss_rule.weights()(i);
            double x = gauss_rule.points()(i);
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
TEST_CASE("QuadratureRule1D map_to_domain", "[quadrature]") 
{
    GaussLegendre<double, 1> gauss_rule(3);
    auto [mapped_points, mapped_weights] = gauss_rule.map_to_domain(0.0, 2.0);
    // Sum of mapped weights should equal the interval length
    REQUIRE(mapped_weights.sum() == Approx(2.0));
    // Points should be in [0, 2]
    for (std::size_t i = 0; i < gauss_rule.num_points(); ++i) {
        REQUIRE(mapped_points(i) >= 0.0);
        REQUIRE(mapped_points(i) <= 2.0);
    }
}

/**
 * Tensor product quadrature rule test.
 */
TEST_CASE("QuadratureRule tensor product", "[quadrature][gauss_legendre]") {
    GaussLegendre<double, 1> gauss_2pt(2);
    GaussLegendre<double, 1> gauss_3pt(3);

    SECTION("Isotropic 2D tensor product") {
        std::array<const QuadratureRule<double, 1>*, 2> rules = {&gauss_3pt, &gauss_3pt};
        auto [points, weights] = QuadratureRule<double, 2>::tensor_product(rules);
        REQUIRE(points.rows() == 9);
        REQUIRE(points.cols() == 2);
        REQUIRE(weights.sum() == Approx(4.0));
    }

    SECTION("Anisotropic 2D tensor product") {
        std::array<const QuadratureRule<double, 1>*, 2> rules = {&gauss_2pt, &gauss_3pt};
        auto [points, weights] = QuadratureRule<double, 2>::tensor_product(rules);
        REQUIRE(points.rows() == 6);
        REQUIRE(points.cols() == 2);
        REQUIRE(weights.sum() == Approx(4.0));
    }

    SECTION("Isotropic 3D tensor product") {
        std::array<const QuadratureRule<double, 1>*, 3> rules = {&gauss_2pt, &gauss_2pt, &gauss_2pt};
        auto [points, weights] = QuadratureRule<double, 3>::tensor_product(rules);
        REQUIRE(points.rows() == 8);
        REQUIRE(points.cols() == 3);
        REQUIRE(weights.sum() == Approx(8.0));
    }
}

/**
 * Multidimensional QuadratureRule class test.
 */
TEST_CASE("Multidimensional QuadratureRule class", "[quadrature][tensor_product]") {
    GaussLegendre<double, 1> gauss_2pt(2);
    GaussLegendre<double, 1> gauss_3pt(3);

    SECTION("2D construction (isotropic)") {
        QuadratureRule<double, 2> tensor_rule(gauss_3pt);
        REQUIRE(tensor_rule.num_points() == 9);
        REQUIRE(tensor_rule.dim() == 2);
        REQUIRE(tensor_rule.weights().sum() == Approx(4.0));
    }

    SECTION("Generalized GaussLegendre<T, d>") {
        GaussLegendre<double, 2> gauss_2d_3pt(3);
        REQUIRE(gauss_2d_3pt.num_points() == 9);
        REQUIRE(gauss_2d_3pt.points().cols() == 2);
        REQUIRE(gauss_2d_3pt.weights().sum() == Approx(4.0));
    }

    SECTION("2D construction (anisotropic)") {
        std::array<const QuadratureRule<double, 1>*, 2> rules = {&gauss_2pt, &gauss_3pt};
        QuadratureRule<double, 2> tensor_rule(rules);
        REQUIRE(tensor_rule.num_points() == 6);
        REQUIRE(tensor_rule.weights().sum() == Approx(4.0));
    }

    SECTION("2D mapping") {
        QuadratureRule<double, 2> tensor_rule(gauss_2pt);
        // Map to [0, 2] x [0, 1]
        auto [points, weights] = tensor_rule.map_to_domain({0.0, 0.0}, {2.0, 1.0});
        REQUIRE(weights.sum() == Approx(2.0)); // Volume = 2*1 = 2
        for (Eigen::Index i = 0; i < points.rows(); ++i) {
            REQUIRE(points(i, 0) >= 0.0);
            REQUIRE(points(i, 0) <= 2.0);
            REQUIRE(points(i, 1) >= 0.0);
            REQUIRE(points(i, 1) <= 1.0);
        }
    }
}

/**
 * Gauss-Legendre 1D integration test.
 */
TEST_CASE("GaussLegendre 1D integration", "[quadrature][gauss_legendre]") {
    SECTION("Polynomial up to degree 5") {
        GaussLegendre<double, 1> gauss_rule(3);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gauss_rule.num_points(); ++i) {
            double x = gauss_rule.points()(i);
            // f(x) = 3x^5 - 2x^4 + x^2 - 7
            integral += gauss_rule.weights()(i) * (3 * std::pow(x, 5) - 2 * std::pow(x, 4) + x * x - 7);
        }
        
        // Analytical integral from -1 to 1:
        // [-2x^4] -> -4/5
        // [x^2] -> 2/3
        // [-7] -> -14
        // Result: -4/5 + 2/3 - 14 = -212/15
        REQUIRE(integral == Approx(-212.0 / 15.0).margin(1e-12));
    }
    
    SECTION("Trigonometric function (sine)") {
        GaussLegendre<double, 1> gauss_rule(10);
        
        double integral = 0.0;
        for (std::size_t i = 0; i < gauss_rule.num_points(); ++i) {
            double x = gauss_rule.points()(i);
            integral += gauss_rule.weights()(i) * std::sin(x);
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
        GaussLegendre<double, 1> gauss_1d(3);
        std::array<const QuadratureRule<double, 1>*, 2> rules = {&gauss_1d, &gauss_1d};
        auto [points, weights] = QuadratureRule<double, 2>::tensor_product(rules);
        
        double integral = 0.0;
        for (Eigen::Index i = 0; i < points.rows(); ++i) {
            double x = points(i, 0);
            double y = points(i, 1);
            integral += weights(i) * (x * x * y * y);
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
        GaussLegendre<double, 1> gauss_1d(3);
        std::array<const QuadratureRule<double, 1>*, 3> rules = {&gauss_1d, &gauss_1d, &gauss_1d};
        auto [points, weights] = QuadratureRule<double, 3>::tensor_product(rules);
        
        double integral = 0.0;
        for (Eigen::Index i = 0; i < points.rows(); ++i) {
            double x = points(i, 0);
            double y = points(i, 1);
            double z = points(i, 2);
            integral += weights(i) * (x * x + y * y + z * z);
        }
        
        // Analytical integral from [-1, 1] x [-1, 1] x [-1, 1]:
        // 3 * (int_{-1}^1 x^2 dx) * (int_{-1}^1 1 dy) * (int_{-1}^1 1 dz)
        // = 3 * (2/3) * 2 * 2 = 8
        REQUIRE(integral == Approx(8.0).margin(1e-12));
    }
}
