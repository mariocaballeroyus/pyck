#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "gauss_legendre.hpp"

using namespace pyck;

TEST_CASE("GaussLegendre functionality", "[quadrature]") {
    SECTION("Zero points") {
        GaussLegendre gl(0);
        REQUIRE(gl.num_points() == 0);
        REQUIRE(gl.dim() == 1);
        REQUIRE(gl.points().rows() == 0);
        REQUIRE(gl.weights().size() == 0);
    }

    SECTION("One point (precomputed)") {
        GaussLegendre gl(1);
        REQUIRE(gl.num_points() == 1);
        REQUIRE(gl.dim() == 1);
        REQUIRE(gl.points()(0, 0) == Approx(0.0));
        REQUIRE(gl.weights()(0) == Approx(2.0));
    }

    SECTION("Two points (precomputed)") {
        GaussLegendre gl(2);
        REQUIRE(gl.num_points() == 2);
        REQUIRE(gl.dim() == 1);
        REQUIRE(gl.points()(0, 0) == Approx(-0.577350269));
        REQUIRE(gl.points()(1, 0) == Approx(0.577350269));
        REQUIRE(gl.weights()(0) == Approx(1.0));
        REQUIRE(gl.weights()(1) == Approx(1.0));
    }

    SECTION("Eight points (precomputed)") {
        GaussLegendre gl(8);
        REQUIRE(gl.num_points() == 8);
        REQUIRE(gl.dim() == 1);
        // Just checking bounds
        REQUIRE(gl.points()(0, 0) < 0);
        REQUIRE(gl.points()(7, 0) > 0);
        REQUIRE(gl.weights().sum() == Approx(2.0));
    }

    SECTION("Computation (10 points)") {
        GaussLegendre gl(10);
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

TEST_CASE("QuadratureRule tensor product", "[quadrature]") {
    GaussLegendre gl1(2);
    GaussLegendre gl2(3);
    
    QuadratureRule prod = gl1.tensor_product(gl2);
    REQUIRE(prod.num_points() == 6);
    REQUIRE(prod.dim() == 2);
    REQUIRE(prod.weights().sum() == Approx(4.0));
}
