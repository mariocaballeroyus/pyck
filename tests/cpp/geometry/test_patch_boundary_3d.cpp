#include "catch.hpp"

#include <array>
#include <cmath>
#include <memory>

#include "patch.hpp"
#include "patch_boundary.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"

using namespace pyck;

namespace
{

/// Build a unit box patch with degree 2, 4 control points per direction.
auto make_unit_box()
{
    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto bu = std::make_shared<BSpline<double>>(2, kv);
    auto bv = std::make_shared<BSpline<double>>(2, kv);
    auto bw = std::make_shared<BSpline<double>>(2, kv);
    return std::make_shared<Patch<double, 3>>(
        box<double>(bu, bv, bw, 1.0, 1.0, 1.0));
}

}  // namespace

// ===========================================================================
// Test 1: PatchBoundary<T, 3> inherits Patch<T, 2> and carries the right
//         number of parent DOFs on each face.
// ===========================================================================

TEST_CASE("PatchBoundary<double, 3>: face construction",
          "[geometry][patch_boundary]")
{
    auto parent = make_unit_box();
    const Index n_per_dir = 4;
    const Index expected_layer_dofs = n_per_dir * n_per_dir;  // 16

    for (std::size_t param_dim = 0; param_dim < 3; ++param_dim) {
        for (bool at_start : {true, false}) {
            auto bdy = create_patch_boundary<double, 3>(parent, param_dim, at_start);

            // PatchBoundary<T, 3> derives from Patch<T, 2>: tdim=2, num_cps = 16.
            CHECK(bdy->tdim() == 2);
            CHECK(bdy->gdim() == 3);
            CHECK(bdy->num_control_pts() == expected_layer_dofs);

            // displacement_dofs is the layer of parent DOFs on this face.
            CHECK(bdy->displacement_dofs().size() == expected_layer_dofs);

            // assembly_dofs returns the parent DOF indices on this face.
            CHECK(bdy->assembly_dofs().size() == expected_layer_dofs);
        }
    }
}


// ===========================================================================
// Test 2: parent_flat_span round-trips against the parent's lexicographic
//         encoding for every boundary span.
// ===========================================================================

TEST_CASE("PatchBoundary<double, 3>: parent_flat_span round-trip",
          "[geometry][patch_boundary]")
{
    auto parent = make_unit_box();
    const auto p_int = parent->tensor_product().num_intervals();

    for (std::size_t param_dim = 0; param_dim < 3; ++param_dim) {
        for (bool at_start : {true, false}) {
            auto bdy = create_patch_boundary<double, 3>(parent, param_dim, at_start);
            const auto b_int = bdy->tensor_product().num_intervals();
            const Index n_bdy_spans = b_int[0] * b_int[1];

            for (Index s = 0; s < n_bdy_spans; ++s) {
                const Index parent_flat = bdy->parent_flat_span(s);

                // Re-decode parent_flat with parent's lexicographic encoding
                // and verify the fixed-direction span and free-direction spans.
                const auto p_spans = parent->decode_span(parent_flat);
                const auto b_spans = bdy->decode_span(s);

                std::size_t bi = 0;
                for (std::size_t k = 0; k < 3; ++k) {
                    if (k == param_dim) {
                        // Fixed direction: must be a valid parent span index.
                        CHECK(p_spans[k] < p_int[k]);
                    } else {
                        // Free direction: must match the boundary's own span.
                        CHECK(p_spans[k] == b_spans[bi]);
                        ++bi;
                    }
                }
            }
        }
    }
}


// ===========================================================================
// Test 3: lift_to_parent puts u_eval_fixed_ into the param_dim_ column and
//         copies the free-direction columns in canonical order.
// ===========================================================================

TEST_CASE("PatchBoundary<double, 3>: lift_to_parent",
          "[geometry][patch_boundary]")
{
    auto parent = make_unit_box();

    ColMatrix<double, 2> input(3, 2);
    input << 0.2, 0.7,
             0.5, 0.5,
             0.9, 0.1;

    for (std::size_t param_dim = 0; param_dim < 3; ++param_dim) {
        for (bool at_start : {true, false}) {
            auto bdy = create_patch_boundary<double, 3>(parent, param_dim, at_start);
            const auto lifted = bdy->lift_to_parent(input);

            REQUIRE(lifted.rows() == 3);
            REQUIRE(lifted.cols() == 3);

            const double expected_fixed = at_start ? 0.0 : 1.0;
            for (Eigen::Index q = 0; q < 3; ++q)
                CHECK(lifted(q, param_dim) == Approx(expected_fixed).margin(1e-14));

            // Free-direction columns must equal the input columns in canonical order.
            std::size_t bi = 0;
            for (std::size_t k = 0; k < 3; ++k) {
                if (k == param_dim) continue;
                for (Eigen::Index q = 0; q < 3; ++q)
                    CHECK(lifted(q, k) == Approx(input(q, bi)).margin(1e-14));
                ++bi;
            }
        }
    }
}


// ===========================================================================
// Test 4: eval_outward_normal on the 6 faces of a unit cube returns ±ê_k
//         aligned with the face's outward direction.
// ===========================================================================

TEST_CASE("PatchBoundary<double, 3>: outward normal on cube faces",
          "[geometry][patch_boundary]")
{
    auto parent = make_unit_box();

    // Reference outward normals for each (param_dim, at_start) combination.
    auto expected_normal = [](std::size_t param_dim, bool at_start) {
        Eigen::Vector3d n = Eigen::Vector3d::Zero();
        n(param_dim) = at_start ? -1.0 : +1.0;
        return n;
    };

    // Pick the first non-degenerate boundary element on each face.
    ColMatrix<double, 2> probes(2, 2);
    probes << 0.30, 0.60,
              0.70, 0.40;

    for (std::size_t param_dim = 0; param_dim < 3; ++param_dim) {
        for (bool at_start : {true, false}) {
            auto bdy = create_patch_boundary<double, 3>(parent, param_dim, at_start);

            // Find the first non-degenerate span on the boundary.
            const auto b_int = bdy->tensor_product().num_intervals();
            const Index n_spans = b_int[0] * b_int[1];
            Index span = 0;
            for (; span < n_spans; ++span) {
                const auto sp = bdy->decode_span(span);
                auto [lu, hu] = bdy->basis(0).knot_vector().span_bounds(sp[0]);
                auto [lv, hv] = bdy->basis(1).knot_vector().span_bounds(sp[1]);
                if (std::abs(hu - lu) > 1e-14 && std::abs(hv - lv) > 1e-14) break;
            }
            REQUIRE(span < n_spans);

            const auto sp = bdy->decode_span(span);
            auto [lu, hu] = bdy->basis(0).knot_vector().span_bounds(sp[0]);
            auto [lv, hv] = bdy->basis(1).knot_vector().span_bounds(sp[1]);

            // Map our probe points into the span's parametric box.
            ColMatrix<double, 2> mapped(2, 2);
            for (int q = 0; q < 2; ++q) {
                mapped(q, 0) = lu + probes(q, 0) * (hu - lu);
                mapped(q, 1) = lv + probes(q, 1) * (hv - lv);
            }

            const auto bbasis = (*bdy).tensor_product().eval_all(mapped, 1);
            const auto bact   = bdy->active_control_pts(span);
            IntrinsicGeometry<double, 2> blocal(bbasis, bact);

            const auto normals = bdy->eval_outward_normal(blocal);
            REQUIRE(normals.rows() == 2);
            REQUIRE(normals.cols() == 3);

            const Eigen::Vector3d expected = expected_normal(param_dim, at_start);
            for (Eigen::Index q = 0; q < 2; ++q) {
                Eigen::Vector3d got = normals.row(q).transpose();
                CHECK(got.norm() == Approx(1.0).margin(1e-12));
                CHECK((got - expected).norm() == Approx(0.0).margin(1e-12));
            }
        }
    }
}
