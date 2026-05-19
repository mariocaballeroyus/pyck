#include "catch.hpp"

#include "knot_vector.hpp"

using namespace pyck;

/**
 * Monotonicity test.
 */
TEST_CASE("KnotVector: monotonicity", "[basis][knots]") {
    for (std::size_t p = 1; p <= 4; ++p) {
        std::size_t n = p + 3;  
        auto kv = KnotVector<double>::clamped_uniform(p, n);

        for (std::size_t i = 0; i + 1 < kv.size(); ++i)
            REQUIRE(kv[i] <= kv[i + 1]);
    }
}

/**
 * Knot span test.
 */
TEST_CASE("KnotVector: find_span", "[basis][knots]") {
    auto kv = KnotVector<double>::clamped_uniform(2, 5);

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
        auto kv = KnotVector<double>::clamped_uniform(p, n);

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
    auto kv = KnotVector<double>::clamped_uniform(p, n);

    std::size_t num_elements = 0;
    for (std::size_t i = 0; i + 1 < kv.size(); ++i) {
        if (kv[i + 1] - kv[i] > 0.0)
            ++num_elements;
    }

    REQUIRE(num_elements == n - p);
}

/**
 * Greville abscissae formula test.
 * Formula: xi_i = 1/p * sum(u_{i+1:i+p+1}) for i = 0, ..., n-1
 */
TEST_CASE("KnotVector: Greville abscissae", "[basis][knots]") {
    std::size_t p = 2;
    std::size_t n = 4;
    auto kv = KnotVector<double>::clamped_uniform(p, n); // [0,0,0, 0.5, 1,1,1]
    
    // Expected: [0, (0+0.5)/2, (0.5+1)/2, 1]
    std::vector<double> expected = {0.0, 0.25, 0.75, 1.0};
    
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 1; j <= p; ++j) {
            sum += kv[i + j];
        }
        REQUIRE(sum / p == Approx(expected[i]));
    }
}

/**
 * Knot insertion: splices values, preserves sortedness.
 */
TEST_CASE("KnotVector: insert", "[basis][knots]") {
    auto kv = KnotVector<double>::clamped_uniform(2, 5); // [0,0,0, 1/3, 2/3, 1,1,1]

    SECTION("single insertion grows the vector by one") {
        auto kv2 = kv.insert_knot(0.5);
        REQUIRE(kv2.size() == kv.size() + 1);
        for (std::size_t i = 0; i + 1 < kv2.size(); ++i)
            REQUIRE(kv2[i] <= kv2[i + 1]);
    }

    SECTION("multiple insertions of the same knot") {
        auto kv3 = kv.insert_knot(0.5).insert_knot(0.5).insert_knot(0.5);
        REQUIRE(kv3.size() == kv.size() + 3);
        // Three consecutive 0.5 values must appear.
        std::size_t run = 0;
        for (std::size_t i = 0; i < kv3.size(); ++i)
            if (kv3[i] == Approx(0.5)) ++run;
        REQUIRE(run == 3);
    }

    SECTION("u outside knot range throws") {
        REQUIRE_THROWS_AS(kv.insert_knot(-0.1), std::invalid_argument);
        REQUIRE_THROWS_AS(kv.insert_knot(1.5), std::invalid_argument);
    }
}
