#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "patch.hpp"
#include "patch_boundary.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knots.hpp"
#include "boundary_field.hpp"
#include "gauss_legendre.hpp"
#include "quadrature.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "penalty_boundary_condition.hpp"
#include "lagrange_coupling_condition.hpp"
#include "dof_layout.hpp"
#include "plane_stress_2d.hpp"
#include "plate_kirchhoff_love_1p.hpp"
#include "assert_conforming.hpp"

using namespace pyck;

// ---------------------------------------------------------------------------
// Helpers (duplicated from test_penalty_coupling_condition.cpp to keep
// each test translation unit self-contained).
// ---------------------------------------------------------------------------
static Ptr<Patch<double, 2>> make_translated_square(
    double L_x, double L_y, Index p, Index n_basis_u, Index n_basis_v,
    double dx = 0.0, double dy = 0.0)
{
    auto bsp_u = std::make_shared<BSpline<double>>(
        p, clamped_uniform_knots<double>(p, n_basis_u));
    auto bsp_v = std::make_shared<BSpline<double>>(
        p, clamped_uniform_knots<double>(p, n_basis_v));
    auto rect = rectangle<double>(bsp_u, bsp_v, L_x, L_y);
    if (dx != 0.0 || dy != 0.0) {
        rect.control_pts().col(0).array() += dx;
        rect.control_pts().col(1).array() += dy;
    }
    return std::make_shared<Patch<double, 2>>(std::move(rect));
}

template <class Patch2D, class GaussRule2D>
static Vector<double> uniform_load_values(
    const Patch2D& patch, const GaussRule2D& gauss, double q0)
{
    Index ns_u = patch.basis(0).knot_vector().num_spans();
    Index ns_v = patch.basis(1).knot_vector().num_spans();
    Index active = 0;
    for (Index e = 0; e < ns_u * ns_v; ++e) {
        Index su = e % ns_u, sv = e / ns_u;
        auto [lou, hiu] = patch.basis(0).knot_vector().span_bounds(su);
        auto [lov, hiv] = patch.basis(1).knot_vector().span_bounds(sv);
        if (std::abs(hiu-lou) > 1e-14 && std::abs(hiv-lov) > 1e-14) ++active;
    }
    return Vector<double>::Constant(active * gauss.num_points(), q0);
}

static double eval_w_at(const Ptr<Patch<double, 2>>& patch,
                        const Vector<double>& u_full,
                        Index dof_offset,
                        Index ndof,
                        double xi, double eta)
{
    auto bu = patch->basis_ptr(0);
    auto bv = patch->basis_ptr(1);
    ColMatrix<double, 2> pt(1, 2);
    pt << xi, eta;
    Index su = bu->find_span(xi);
    Index sv = bv->find_span(eta);
    Index ni_u = patch->tensor_product().num_intervals()[0];
    Index flat = su + sv * ni_u;

    auto [sf, jac] = patch->eval_shape_functions(pt, flat, 0);
    auto active = patch->dof_mapper().get_element_dofs(flat);
    Vector<double> w_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        w_active(i) = u_full(dof_offset + active[i] * ndof + 0);
    return (sf[0] * w_active)(0, 0);
}

// ===========================================================================
// TEST 1 — Stiffness matrix is symmetric (saddle-point block built correctly)
// ===========================================================================
TEST_CASE("LagrangeCouplingCondition: stiffness is symmetric",
          "[conditions][lagrange_coupling]")
{
    const double L = 1.0;
    const Index  p = 3, n = 5;

    auto patch_A = make_translated_square(L, L, p, n, n, /*dx=*/0.0);
    auto patch_B = make_translated_square(L, L, p, n, n, /*dx=*/L);

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.01);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    GaussLegendre<double, 1> g1d(p + 1);

    auto side_a = create_patch_boundary<double, 2>(patch_A, 0, false); // A.u1
    auto side_b = create_patch_boundary<double, 2>(patch_B, 0, true);  // B.u0

    REQUIRE_NOTHROW((assert_conforming<double, 2>(*side_a, *side_b, false)));

    LagrangeCouplingCondition<double, 2> cc(
        *side_a, *side_b, /*idx_a=*/0, /*idx_b=*/1,
        *element, *element, g1d, /*reverse=*/false);

    cc.add(std::make_shared<BasisValue<double>>(0),
           std::make_shared<BasisValue<double>>(0),
           /*sign_b=*/+1.0);
    cc.add(std::make_shared<BasisNormalSlope<double>>(0),
           std::make_shared<BasisNormalSlope<double>>(0),
           /*sign_b=*/-1.0);

    DofLayout layout;
    const Index ndof = element->num_node_dofs();
    auto block_a = layout.allocate(DofType::Primal, patch_A->num_control_pts() * ndof, ndof);
    auto block_b = layout.allocate(DofType::Primal, patch_B->num_control_pts() * ndof, ndof);
    std::vector<DofLayout::BlockId> blocks{block_a, block_b};

    cc.allocate_dofs(layout, blocks);

    const Index N_total = layout.num_dofs();
    Matrix<double> K = Matrix<double>::Zero(N_total, N_total);
    Vector<double> F = Vector<double>::Zero(N_total);

    cc.apply(K, F, layout, blocks);

    REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());
    REQUIRE(K.norm() > 0.0);
}

// ===========================================================================
// TEST 2 — num_dofs() and allocated multiplier count
// ===========================================================================
TEST_CASE("LagrangeCouplingCondition: num_dofs reports total multipliers",
          "[conditions][lagrange_coupling]")
{
    const double L = 1.0;
    const Index  p = 2, n = 4;

    auto patch_A = make_translated_square(L, L, p, n, n, 0.0);
    auto patch_B = make_translated_square(L, L, p, n, n, L);

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.01);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    GaussLegendre<double, 1> g1d(p + 1);

    auto side_a = create_patch_boundary<double, 2>(patch_A, 0, false);
    auto side_b = create_patch_boundary<double, 2>(patch_B, 0, true);

    LagrangeCouplingCondition<double, 2> cc(
        *side_a, *side_b, 0, 1, *element, *element, g1d);

    REQUIRE(cc.num_dofs() == 0);  // no terms → no multipliers

    cc.add(std::make_shared<BasisValue<double>>(0),
           std::make_shared<BasisValue<double>>(0));
    REQUIRE(cc.num_dofs() == static_cast<std::size_t>(side_a->num_control_pts()));

    cc.add(std::make_shared<BasisNormalSlope<double>>(0),
           std::make_shared<BasisNormalSlope<double>>(0));
    REQUIRE(cc.num_dofs() == 2 * static_cast<std::size_t>(side_a->num_control_pts()));

    DofLayout layout;
    const Index ndof = element->num_node_dofs();
    layout.allocate(DofType::Primal, patch_A->num_control_pts() * ndof, ndof);
    layout.allocate(DofType::Primal, patch_B->num_control_pts() * ndof, ndof);
    const Index n_primal = layout.num_dofs();
    std::vector<DofLayout::BlockId> blocks{0, 1};

    cc.allocate_dofs(layout, blocks);
    REQUIRE(layout.num_dofs() == n_primal + static_cast<Index>(cc.num_dofs()));
}

// ===========================================================================
// TEST 3 — Construction throws for null fields and equal patch indices
// ===========================================================================
TEST_CASE("LagrangeCouplingCondition: invalid configurations throw",
          "[conditions][lagrange_coupling]")
{
    const double L = 1.0;
    const Index  p = 2, n = 4;

    auto patch_A = make_translated_square(L, L, p, n, n, 0.0);
    auto patch_B = make_translated_square(L, L, p, n, n, L);

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.01);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    GaussLegendre<double, 1> g1d(p + 1);

    auto side_a = create_patch_boundary<double, 2>(patch_A, 0, false);
    auto side_b = create_patch_boundary<double, 2>(patch_B, 0, true);

    REQUIRE_THROWS_AS(
        (LagrangeCouplingCondition<double, 2>(
            *side_a, *side_b, /*idx_a=*/2, /*idx_b=*/2,
            *element, *element, g1d)),
        std::invalid_argument);

    LagrangeCouplingCondition<double, 2> cc(
        *side_a, *side_b, 0, 1, *element, *element, g1d);
    REQUIRE_THROWS_AS(
        cc.add(nullptr, std::make_shared<BasisValue<double>>(0)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cc.add(std::make_shared<BasisValue<double>>(0), nullptr),
        std::invalid_argument);
}

// ===========================================================================
// TEST 4 — Two-patch reproduces single-patch reference (clamped plate)
//
// Same setup as test_penalty_coupling_condition.cpp TEST 5, but with the
// inter-patch w-coupling enforced via Lagrange multipliers instead of a
// penalty. The resulting saddle-point system is solved with a partial-pivot
// LU; deflection at interior physical points should match the single-patch
// reference within the same 5% tolerance.
// ===========================================================================
TEST_CASE("LagrangeCouplingCondition: two-patch reproduces single-patch",
          "[conditions][lagrange_coupling][physics]")
{
    const double E = 1.0e7, nu = 0.3, h = 0.01;
    const double L = 1.0, q0 = -1.0;
    const Index  p = 3, N = 8;
    const double D = E * h*h*h / (12.0 * (1.0 - nu*nu));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);
    GaussLegendre<double, 1> g1d(p + 1);

    // ===== Single-patch reference: 2L x L =====
    auto patch_ref = make_translated_square(2.0 * L, L, p, 2 * N, N);
    Vector<double> load_ref = uniform_load_values(*patch_ref, *gauss2d, q0);

    LinearElasticProblem<double, 2> prob_ref(patch_ref, element, gauss2d);
    prob_ref.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *patch_ref, *element, *gauss2d, load_ref));

    const double alpha_bc = 1e8 * D / (L * L * L);
    std::vector<Ptr<PatchBoundary<double, 2>>> bds_ref;
    for (std::size_t dim = 0; dim < 2; ++dim)
        for (bool start : {true, false}) {
            bds_ref.push_back(create_patch_boundary<double, 2>(patch_ref, dim, start));
            auto pen = std::make_shared<PenaltyBoundaryCondition<double, 2>>(
                *bds_ref.back(), *element, g1d);
            pen->add(std::make_shared<TransverseDisplacement<double>>(), alpha_bc, 0.0);
            prob_ref.add_condition(pen);
        }

    Matrix<double> K_ref;  Vector<double> F_ref;
    prob_ref.assemble(K_ref, F_ref);
    Eigen::PartialPivLU<Matrix<double>> solver_ref(K_ref);
    Vector<double> u_ref = solver_ref.solve(F_ref);

    const double w_ref_left  = eval_w_at(patch_ref, u_ref, 0, 1, 0.25, 0.5);
    const double w_ref_right = eval_w_at(patch_ref, u_ref, 0, 1, 0.75, 0.5);

    // ===== Two-patch coupled with Lagrange multipliers =====
    auto patch_A = make_translated_square(L, L, p, N, N, 0.0);
    auto patch_B = make_translated_square(L, L, p, N, N, L);

    LinearElasticProblem<double, 2> prob;
    auto idx_A = prob.add_patch(patch_A, element, gauss2d);
    auto idx_B = prob.add_patch(patch_B, element, gauss2d);

    Vector<double> load_A = uniform_load_values(*patch_A, *gauss2d, q0);
    Vector<double> load_B = uniform_load_values(*patch_B, *gauss2d, q0);
    prob.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *patch_A, *element, *gauss2d, load_A), idx_A);
    prob.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *patch_B, *element, *gauss2d, load_B), idx_B);

    auto clamp_outer = [&](const Ptr<Patch<double, 2>>& p, std::size_t patch_idx,
                            std::size_t skip_dim, bool skip_start,
                            std::vector<Ptr<PatchBoundary<double, 2>>>& store) {
        for (std::size_t dim = 0; dim < 2; ++dim)
            for (bool start : {true, false}) {
                if (dim == skip_dim && start == skip_start) continue;
                store.push_back(create_patch_boundary<double, 2>(p, dim, start));
                auto pen = std::make_shared<PenaltyBoundaryCondition<double, 2>>(
                    *store.back(), *element, g1d);
                pen->add(std::make_shared<TransverseDisplacement<double>>(),
                         alpha_bc, 0.0);
                prob.add_condition(pen, patch_idx);
            }
    };

    std::vector<Ptr<PatchBoundary<double, 2>>> bds;
    clamp_outer(patch_A, idx_A, 0, false, bds);
    clamp_outer(patch_B, idx_B, 0, true,  bds);

    auto side_a = create_patch_boundary<double, 2>(patch_A, 0, false); // A.u1
    auto side_b = create_patch_boundary<double, 2>(patch_B, 0, true);  // B.u0
    REQUIRE_NOTHROW((assert_conforming<double, 2>(*side_a, *side_b, false)));

    auto cc = std::make_shared<LagrangeCouplingCondition<double, 2>>(
        *side_a, *side_b, idx_A, idx_B, *element, *element, g1d);
    cc->add(std::make_shared<BasisValue<double>>(0),
            std::make_shared<BasisValue<double>>(0),
            /*sign_b=*/+1.0);
    prob.add_condition(cc, idx_A);

    Matrix<double> K;  Vector<double> F;
    prob.assemble(K, F);

    REQUIRE((K - K.transpose()).norm() < 1e-10 * K.norm());

    Eigen::PartialPivLU<Matrix<double>> solver(K);
    Vector<double> u = solver.solve(F);

    const Index n_dof_A = patch_A->num_control_pts();

    double w_two_left  = eval_w_at(patch_A, u, 0,        1, 0.5, 0.5);
    double w_two_right = eval_w_at(patch_B, u, n_dof_A,  1, 0.5, 0.5);

    INFO("w_ref_left = " << w_ref_left << "  w_two_left = " << w_two_left);
    INFO("w_ref_right = " << w_ref_right << "  w_two_right = " << w_two_right);

    REQUIRE(std::abs(w_two_left  - w_ref_left)  < 0.05 * std::abs(w_ref_left));
    REQUIRE(std::abs(w_two_right - w_ref_right) < 0.05 * std::abs(w_ref_right));
}

// ===========================================================================
// TEST 5 — Constraint is enforced exactly at boundary control points
//
// With BasisValue(0) Lagrange coupling, the discrete constraint
//   ∫_Γ N_λ_i (w_A - sB·w_B) dΓ = 0   ∀i
// combined with conformity of the boundary basis between sides A and B
// implies w_A(cp_i) = sB·w_B(cp_i) at every shared control point on the
// interface. Verify that residual to machine precision after solving.
// ===========================================================================
TEST_CASE("LagrangeCouplingCondition: enforces continuity at shared control pts",
          "[conditions][lagrange_coupling][physics]")
{
    const double E = 1.0e7, nu = 0.3, h = 0.01;
    const double L = 1.0, q0 = -1.0;
    const Index  p = 3, N = 6;
    const double D = E * h*h*h / (12.0 * (1.0 - nu*nu));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);
    GaussLegendre<double, 1> g1d(p + 1);

    auto patch_A = make_translated_square(L, L, p, N, N, 0.0);
    auto patch_B = make_translated_square(L, L, p, N, N, L);

    LinearElasticProblem<double, 2> prob;
    auto idx_A = prob.add_patch(patch_A, element, gauss2d);
    auto idx_B = prob.add_patch(patch_B, element, gauss2d);

    Vector<double> load_A = uniform_load_values(*patch_A, *gauss2d, q0);
    Vector<double> load_B = uniform_load_values(*patch_B, *gauss2d, q0);
    prob.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *patch_A, *element, *gauss2d, load_A), idx_A);
    prob.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *patch_B, *element, *gauss2d, load_B), idx_B);

    const double alpha_bc = 1e8 * D / (L * L * L);
    std::vector<Ptr<PatchBoundary<double, 2>>> bds;
    auto clamp_outer = [&](const Ptr<Patch<double, 2>>& p, std::size_t patch_idx,
                            std::size_t skip_dim, bool skip_start) {
        for (std::size_t dim = 0; dim < 2; ++dim)
            for (bool start : {true, false}) {
                if (dim == skip_dim && start == skip_start) continue;
                bds.push_back(create_patch_boundary<double, 2>(p, dim, start));
                auto pen = std::make_shared<PenaltyBoundaryCondition<double, 2>>(
                    *bds.back(), *element, g1d);
                pen->add(std::make_shared<TransverseDisplacement<double>>(),
                         alpha_bc, 0.0);
                prob.add_condition(pen, patch_idx);
            }
    };
    clamp_outer(patch_A, idx_A, 0, false);
    clamp_outer(patch_B, idx_B, 0, true);

    auto side_a = create_patch_boundary<double, 2>(patch_A, 0, false);
    auto side_b = create_patch_boundary<double, 2>(patch_B, 0, true);

    auto cc = std::make_shared<LagrangeCouplingCondition<double, 2>>(
        *side_a, *side_b, idx_A, idx_B, *element, *element, g1d);
    cc->add(std::make_shared<BasisValue<double>>(0),
            std::make_shared<BasisValue<double>>(0),
            /*sign_b=*/+1.0);
    prob.add_condition(cc, idx_A);

    Matrix<double> K;  Vector<double> F;
    prob.assemble(K, F);
    Eigen::PartialPivLU<Matrix<double>> solver(K);
    Vector<double> u = solver.solve(F);

    // Sample w on side A (xi=1, eta sweep) and side B (xi=0, eta sweep).
    const Index n_dof_A = patch_A->num_control_pts();
    const Index n_eta = 21;
    double max_jump = 0.0;
    double max_w    = 0.0;
    for (Index k = 0; k < n_eta; ++k) {
        const double eta = static_cast<double>(k) / (n_eta - 1);
        const double wA = eval_w_at(patch_A, u, 0,       1, 1.0, eta);
        const double wB = eval_w_at(patch_B, u, n_dof_A, 1, 0.0, eta);
        max_jump = std::max(max_jump, std::abs(wA - wB));
        max_w    = std::max(max_w,    std::abs(wA));
    }

    INFO("max |wA - wB| = " << max_jump << ", max |wA| = " << max_w);
    REQUIRE(max_jump < 1e-10 * std::max(max_w, 1.0));
}
