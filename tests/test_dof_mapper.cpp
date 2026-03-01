#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "dof_mapper.hpp"

using namespace pyck;

TEST_CASE("DofMapper 1D functionality", "[geometry][dof_mapper]") {
    DofMapper<1> mapper({5});

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
    DofMapper<2> mapper({5, 4});

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
    DofMapper<2> mapper({3, 3});
    REQUIRE_THROWS_AS(mapper.get_boundary_dofs(2, true), std::invalid_argument);
}
