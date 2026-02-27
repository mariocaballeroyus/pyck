#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "knots.hpp"

using namespace pyck;

/**
 * Monotonicity test.
 *
 * A valid knot vector must be non-decreasing: ξ_i ≤ ξ_{i+1} for all i.
 * We verify this property on vectors produced by the clamped_uniform factory
 * for several polynomial degrees.
 */
TEST_CASE("KnotVector: monotonicity", "[basis][knots]") {
    for (std::size_t p = 1; p <= 4; ++p) {
        std::size_t n = p + 3;  // a few more basis functions than the minimum
        auto kv = KnotVector::clamped_uniform(p, n);

        for (std::size_t i = 0; i + 1 < kv.size(); ++i)
            REQUIRE(kv[i] <= kv[i + 1]);
    }
}

/**
 * Knot span test.
 *
 * find_span(p, u) must return the index i such that ξ_i ≤ u < ξ_{i+1}.
 * We test interior points and the critical edge case u = ξ_max = 1.0,
 * which must return the last valid span index n (the index of the last
 * non-trivial basis function).
 */
TEST_CASE("KnotVector: find_span", "[basis][knots]") {
    // Degree 2, 5 basis → knots = {0,0,0, 1/3, 2/3, 1,1,1}
    auto kv = KnotVector::clamped_uniform(2, 5);

    SECTION("interior points land in the correct span") {
        // u = 0.0 → first active span = index p = 2
        REQUIRE(kv.find_span(2, 0.0) == 2);

        // u = 0.5 lies in [1/3, 2/3) → span index 3
        REQUIRE(kv.find_span(2, 0.5) == 3);
    }

    SECTION("u = ξ_max returns the last valid span") {
        // Last valid span index n = num_knots - p - 2 = 8 - 2 - 2 = 4
        REQUIRE(kv.find_span(2, 1.0) == 4);
    }
}

/**
 * Clamped property test.
 *
 * For a clamped knot vector on [0, 1] of degree p, the first p+1 knots must
 * all equal 0.0 and the last p+1 knots must all equal 1.0.
 */
TEST_CASE("KnotVector: clamped property", "[basis][knots]") {
    for (std::size_t p = 1; p <= 4; ++p) {
        std::size_t n = p + 2;
        auto kv = KnotVector::clamped_uniform(p, n);

        // First p+1 knots
        for (std::size_t i = 0; i <= p; ++i)
            REQUIRE(kv[i] == Approx(0.0));

        // Last p+1 knots
        for (std::size_t i = kv.size() - p - 1; i < kv.size(); ++i)
            REQUIRE(kv[i] == Approx(1.0));
    }
}

/**
 * Unique spans test.
 *
 * The number of non-zero-length knot spans (elements where ξ_{i+1} - ξ_i > 0)
 * must match the expected number of mesh elements.
 *
 * For a clamped uniform vector with n basis functions of degree p,
 * the number of elements is n - p.
 */
TEST_CASE("KnotVector: unique spans", "[basis][knots]") {
    std::size_t p = 2, n = 6;
    auto kv = KnotVector::clamped_uniform(p, n);

    std::size_t num_elements = 0;
    for (std::size_t i = 0; i + 1 < kv.size(); ++i) {
        if (kv[i + 1] - kv[i] > 0.0)
            ++num_elements;
    }

    // Expected: n - p = 6 - 2 = 4 elements
    REQUIRE(num_elements == n - p);
}
