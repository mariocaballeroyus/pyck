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
        auto layer0 = mapper.get_layer_dofs(0, true, 0);
        auto layer1 = mapper.get_layer_dofs(0, true, 1);
        REQUIRE(layer0.size() == 1);
        REQUIRE(layer1.size() == 1);
        REQUIRE(layer0[0] == 0);
        REQUIRE(layer1[0] == 1);
    }

    SECTION("Boundary DOFs at End (u=1)") {
        auto layer0 = mapper.get_layer_dofs(0, false, 0);
        auto layer1 = mapper.get_layer_dofs(0, false, 1);
        REQUIRE(layer0.size() == 1);
        REQUIRE(layer1.size() == 1);
        REQUIRE(layer0[0] == 4);
        REQUIRE(layer1[0] == 3);
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
        auto layer0 = mapper.get_layer_dofs(0, true, 0);
        auto layer1 = mapper.get_layer_dofs(0, true, 1);
        REQUIRE(layer0.size() == 4);
        REQUIRE(layer1.size() == 4);
        
        // layer 0: (0,0)=0, (0,1)=5, (0,2)=10, (0,3)=15
        REQUIRE(layer0[0] == 0);
        REQUIRE(layer0[1] == 5);
        REQUIRE(layer0[2] == 10);
        REQUIRE(layer0[3] == 15);

        // layer 1: (1,0)=1, (1,1)=6, (1,2)=11, (1,3)=16
        REQUIRE(layer1[0] == 1);
        REQUIRE(layer1[1] == 6);
        REQUIRE(layer1[2] == 11);
        REQUIRE(layer1[3] == 16);
    }

    SECTION("Boundary DOFs at End u=1") {
        auto layer0 = mapper.get_layer_dofs(0, false, 0);
        auto layer1 = mapper.get_layer_dofs(0, false, 1);
        REQUIRE(layer0.size() == 4);
        REQUIRE(layer1.size() == 4);
        
        // layer 0: i=4, j=0..3 -> IDs: 4, 9, 14, 19
        REQUIRE(layer0[0] == 4);
        REQUIRE(layer0[1] == 9);
        REQUIRE(layer0[2] == 14);
        REQUIRE(layer0[3] == 19);

        // layer 1: i=3, j=0..3 -> IDs: 3, 8, 13, 18
        REQUIRE(layer1[0] == 3);
        REQUIRE(layer1[1] == 8);
        REQUIRE(layer1[2] == 13);
        REQUIRE(layer1[3] == 18);
    }

    SECTION("Boundary DOFs at Start v=0") {
        auto layer0 = mapper.get_layer_dofs(1, true, 0);
        auto layer1 = mapper.get_layer_dofs(1, true, 1);
        REQUIRE(layer0.size() == 5);
        REQUIRE(layer1.size() == 5);
        
        // layer 0: j=0, i=0..4 -> IDs: 0, 1, 2, 3, 4
        for (int i=0; i<5; ++i) REQUIRE(layer0[i] == i);
        // layer 1: j=1, i=0..4 -> IDs: 5, 6, 7, 8, 9
        for (int i=0; i<5; ++i) REQUIRE(layer1[i] == 5+i);
    }
}

TEST_CASE("DofMapper bounds checking", "[geometry][dof_mapper]") {
    DofMapper<2> mapper({3, 3}, {2, 2});
    REQUIRE_THROWS_AS(mapper.get_layer_dofs(2, true, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(mapper.get_layer_dofs(0, true, 3), std::invalid_argument);
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
