#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "dof_mapper.hpp"

using namespace pyck;

TEST_CASE("DofMapper 1D functionality", "[geometry][dof_mapper]") {
    DofMapper<1> mapper({5}, {2});

    SECTION("Global Index Mapping") {
        REQUIRE(mapper.to_global({0}) == 0);
        REQUIRE(mapper.to_global({4}) == 4);
    }

    SECTION("Boundary DOFs at Start (u=0)") {
        auto bnd_dofs = mapper.get_boundary_dofs(0, true);
        REQUIRE(bnd_dofs.size() == 2);
        REQUIRE(bnd_dofs[0] == 0);
        REQUIRE(bnd_dofs[1] == 1);
    }

    SECTION("Boundary DOFs at End (u=1)") {
        auto bnd_dofs = mapper.get_boundary_dofs(0, false);
        REQUIRE(bnd_dofs.size() == 2);
        REQUIRE(bnd_dofs[0] == 3);
        REQUIRE(bnd_dofs[1] == 4);
    }
}

TEST_CASE("DofMapper 2D functionality", "[geometry][dof_mapper]") {
    // 5 basis funcs in u-direction, 4 basis funcs in v-direction
    DofMapper<2> mapper({5, 4}, {2, 2});

    SECTION("Global Index Mapping") {
        // ID = i + j * Nu = i + j * 5
        REQUIRE(mapper.to_global({0, 0}) == 0);
        REQUIRE(mapper.to_global({1, 0}) == 1);
        REQUIRE(mapper.to_global({0, 1}) == 5);
        REQUIRE(mapper.to_global({2, 3}) == 2 + 3 * 5); // 17
        REQUIRE(mapper.to_global({4, 3}) == 4 + 3 * 5); // 19 (last element)
    }

    SECTION("Boundary DOFs at Start u=0") {
        auto bnd_dofs = mapper.get_boundary_dofs(0, true);
        // There should be 2 layers along the u-boundary, for all 4 v-indices: 2 * 4 = 8
        REQUIRE(bnd_dofs.size() == 8);
        // j=0 -> i=0, i=1 -> IDs: 0, 1
        // j=1 -> i=0, i=1 -> IDs: 5, 6
        // j=2 -> i=0, i=1 -> IDs: 10, 11
        // j=3 -> i=0, i=1 -> IDs: 15, 16
        REQUIRE(bnd_dofs[0] == 0);
        REQUIRE(bnd_dofs[1] == 1);
        REQUIRE(bnd_dofs[2] == 5);
        REQUIRE(bnd_dofs[3] == 6);
        REQUIRE(bnd_dofs[4] == 10);
        REQUIRE(bnd_dofs[5] == 11);
        REQUIRE(bnd_dofs[6] == 15);
        REQUIRE(bnd_dofs[7] == 16);
    }

    SECTION("Boundary DOFs at End u=1") {
        auto bnd_dofs = mapper.get_boundary_dofs(0, false);
        // Last 2 layers: i=3, i=4 for all j
        REQUIRE(bnd_dofs.size() == 8);
        // j=0 -> i=3, i=4 -> IDs: 3, 4
        // j=1 -> i=3, i=4 -> IDs: 8, 9
        // j=2 -> i=3, i=4 -> IDs: 13, 14
        // j=3 -> i=3, i=4 -> IDs: 18, 19
        REQUIRE(bnd_dofs[0] == 3);
        REQUIRE(bnd_dofs[1] == 4);
        REQUIRE(bnd_dofs[4] == 13);
        REQUIRE(bnd_dofs[5] == 14);
        REQUIRE(bnd_dofs[7] == 19);
    }

    SECTION("Boundary DOFs at Start v=0") {
        auto bnd_dofs = mapper.get_boundary_dofs(1, true);
        // 2 layers along v-boundary (j=0, j=1), for all 5 u-indices: 5 * 2 = 10
        REQUIRE(bnd_dofs.size() == 10);
        // i=0..4, j=0, 1
        // Order of traversal is i in outer dim, j in param dim
        // Actually traversal goes i=0..4 and j=0,1 inside the recursion logic:
        // iterate does current_dim=0 (5 loops) -> current_dim=1 (2 loops).
        // Since j is param_dim=1, for each i: it does j=0, j=1
        // i=0: {0,0}=0, {0,1}=5
        // i=1: {1,0}=1, {1,1}=6
        // ...
        REQUIRE(bnd_dofs[0] == 0);
        REQUIRE(bnd_dofs[1] == 1);
        REQUIRE(bnd_dofs[5] == 5);
        REQUIRE(bnd_dofs[9] == 9);
    }
}

TEST_CASE("DofMapper bounds checking", "[geometry][dof_mapper]") {
    DofMapper<2> mapper({3, 3}, {2, 2});
    REQUIRE_THROWS_AS(mapper.get_boundary_dofs(2, true), std::invalid_argument);
}

TEST_CASE("DofMapper 1D get_element_dofs", "[geometry][dof_mapper]") {
    // degree=2, 5 basis functions -> 7 knot spans (intervals)
    // Clamped knot vector: {0, 0, 0, 0.33, 0.67, 1, 1, 1}
    // Span 0: [0, 0)     zero-length
    // Span 1: [0, 0)     zero-length
    // Span 2: [0, 0.33)  active basis: 0, 1, 2
    // Span 3: [0.33,0.67) active basis: 1, 2, 3
    // Span 4: [0.67,1)   active basis: 2, 3, 4
    // Span 5: [1, 1)     zero-length
    // Span 6: [1, 1)     zero-length
    DofMapper<1> mapper({5}, {2});

    SECTION("Interior element (span 2)") {
        auto dofs = mapper.get_element_dofs(2);
        REQUIRE(dofs.size() == 3);
        REQUIRE(dofs[0] == 0);
        REQUIRE(dofs[1] == 1);
        REQUIRE(dofs[2] == 2);
    }

    SECTION("Interior element (span 3)") {
        auto dofs = mapper.get_element_dofs(3);
        REQUIRE(dofs.size() == 3);
        REQUIRE(dofs[0] == 1);
        REQUIRE(dofs[1] == 2);
        REQUIRE(dofs[2] == 3);
    }

    SECTION("Interior element (span 4)") {
        auto dofs = mapper.get_element_dofs(4);
        REQUIRE(dofs.size() == 3);
        REQUIRE(dofs[0] == 2);
        REQUIRE(dofs[1] == 3);
        REQUIRE(dofs[2] == 4);
    }

    SECTION("Zero-length span at start (span 0)") {
        // span=0, degree=2 -> active = max(0,0-2)..min(0,4) = 0..0
        auto dofs = mapper.get_element_dofs(0);
        REQUIRE(dofs.size() == 1);
        REQUIRE(dofs[0] == 0);
    }

    SECTION("Out of bounds element") {
        REQUIRE_THROWS_AS(mapper.get_element_dofs(7), std::invalid_argument);
    }
}

TEST_CASE("DofMapper 2D get_element_dofs", "[geometry][dof_mapper]") {
    // 3 basis funcs in u (degree 1), 3 basis funcs in v (degree 1)
    // intervals per direction = 3 + 1 = 4
    // Total elements = 16
    // Global DOF = i + j * 3
    //
    //  j=2: 6  7  8
    //  j=1: 3  4  5
    //  j=0: 0  1  2
    //       i=0 i=1 i=2
    DofMapper<2> mapper({3, 3}, {1, 1});

    SECTION("First non-zero span (1,1)") {
        // elem_idx for span (1,1): 1*4 + 1 = 5
        // u-span=1, degree=1: active u-basis = {0, 1}
        // v-span=1, degree=1: active v-basis = {0, 1}
        // DOFs in tensor-product order (u outer, v inner):
        //   (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4
        auto dofs = mapper.get_element_dofs(5);
        REQUIRE(dofs.size() == 4);
        REQUIRE(dofs[0] == 0);
        REQUIRE(dofs[1] == 3);
        REQUIRE(dofs[2] == 1);
        REQUIRE(dofs[3] == 4);
    }

    SECTION("Element (2,2)") {
        // elem_idx for span (2,2): 2*4 + 2 = 10
        // u-span=2, degree=1: active u-basis = {1, 2}
        // v-span=2, degree=1: active v-basis = {1, 2}
        // DOFs in tensor-product order (u outer, v inner):
        //   (1,1)=4, (1,2)=7, (2,1)=5, (2,2)=8
        auto dofs = mapper.get_element_dofs(10);
        REQUIRE(dofs.size() == 4);
        REQUIRE(dofs[0] == 4);
        REQUIRE(dofs[1] == 7);
        REQUIRE(dofs[2] == 5);
        REQUIRE(dofs[3] == 8);
    }
}
