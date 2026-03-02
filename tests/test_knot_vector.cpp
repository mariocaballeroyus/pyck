#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "knots.hpp"

using namespace pyck;

/**
 * Monotonicity test.
 */
TEST_CASE("KnotVector: monotonicity", "[basis][knots]") {
    for (std::size_t p = 1; p <= 4; ++p) {
        std::size_t n = p + 3;  
        auto kv = clamped_uniform_knots<double>(p, n);

        for (std::size_t i = 0; i + 1 < kv.size(); ++i)
            REQUIRE(kv[i] <= kv[i + 1]);
    }
}

/**
 * Knot span test.
 */
TEST_CASE("KnotVector: find_span", "[basis][knots]") {
    auto kv = clamped_uniform_knots<double>(2, 5);

    SECTION("interior points land in the correct span") {
        REQUIRE(kv.find_span(2, 0.0) == 2);
        REQUIRE(kv.find_span(2, 0.5) == 3);
    }

    SECTION("u = ξ_max returns the last valid span") {
        REQUIRE(kv.find_span(2, 1.0) == 4);
    }
}

/**
 * Clamped property test.
 */
TEST_CASE("KnotVector: clamped property", "[basis][knots]") {
    for (std::size_t p = 1; p <= 4; ++p) {
        std::size_t n = p + 2;
        auto kv = clamped_uniform_knots<double>(p, n);

        for (std::size_t i = 0; i <= p; ++i)
            REQUIRE(kv[i] == Approx(0.0));

        for (std::size_t i = kv.size() - p - 1; i < kv.size(); ++i)
            REQUIRE(kv[i] == Approx(1.0));
    }
}

/**
 * Unique spans test.
 */
TEST_CASE("KnotVector: unique spans", "[basis][knots]") {
    std::size_t p = 2, n = 6;
    auto kv = clamped_uniform_knots<double>(p, n);

    std::size_t num_elements = 0;
    for (std::size_t i = 0; i + 1 < kv.size(); ++i) {
        if (kv[i + 1] - kv[i] > 0.0)
            ++num_elements;
    }

    REQUIRE(num_elements == n - p);
}
