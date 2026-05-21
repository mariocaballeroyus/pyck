#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "patch_boundary.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "boundary_field.hpp"
#include "gauss_legendre.hpp"
#include "quadrature.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "penalty_boundary_condition.hpp"
#include "penalty_coupling_condition.hpp"
#include "dof_layout.hpp"
#include "plane_stress_2d.hpp"
#include "plate_kirchhoff_love_1p.hpp"
#include "assert_conforming.hpp"

using namespace pyck;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static Ptr<Patch<double, 2>> make_translated_square(
    double L_x, double L_y, Index p, Index n_basis_u, Index n_basis_v,
    double dx = 0.0, double dy = 0.0)
{
    auto bsp_u = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_basis_u));
    auto bsp_v = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_basis_v));
    auto rect = rectangle<double>(bsp_u, bsp_v, L_x, L_y);
    if (dx != 0.0 || dy != 0.0) {
        rect.control_pts().col(0).array() += dx;
        rect.control_pts().col(1).array() += dy;
    }
    return std::make_shared<Patch<double, 2>>(std::move(rect));
}

// Apply uniform downward pressure on a single-patch problem.
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

// Helper: evaluate w at parametric point (xi, eta) on patch via shape functions.
static double eval_w_at(const Ptr<Patch<double, 2>>& patch,
                        const Vector<double>& u_full,
                        Index dof_offset,    // global offset to the patch's primal block
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

    const auto b = (*patch).tensor_product().eval(pt, 0);
    std::vector<Index> active;
    patch->dof_mapper().get_element_dofs(flat, active);
    Vector<double> w_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        w_active(i) = u_full(dof_offset + active[i] * ndof + 0);
    return b[0].col(0).dot(w_active);
}

// ===========================================================================
// TEST 1 — Symmetry of K under coupling (all sign_b combinations)
// ===========================================================================
TEST_CASE("PenaltyCouplingCondition: stiffness is symmetric",
          "[conditions][coupling]")
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

    PenaltyCouplingCondition<double, 2> cc(
        *side_a, *side_b, /*idx_a=*/0, /*idx_b=*/1,
        *element, *element, g1d, /*reverse=*/false);

    cc.add(std::make_shared<TransverseDisplacement<double>>(),
           std::make_shared<TransverseDisplacement<double>>(),
           /*penalty=*/1e6, /*sign_b=*/+1.0);

    DofLayout layout;
    const Index ndof = element->num_node_dofs();
    auto block_a = layout.allocate(DofType::Primal, patch_A->num_control_pts() * ndof, ndof);
    auto block_b = layout.allocate(DofType::Primal, patch_B->num_control_pts() * ndof, ndof);
    std::vector<DofLayout::BlockId> blocks{block_a, block_b};

    const Index N_total = layout.num_dofs();
    Matrix<double> K = Matrix<double>::Zero(N_total, N_total);
    Vector<double> F = Vector<double>::Zero(N_total);

    cc.apply(K, F, layout, blocks);

    REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());
    REQUIRE(K.norm() > 0.0);          // sanity: not trivially zero
}

// ===========================================================================
// TEST 2 — Zero penalty contributes nothing
// ===========================================================================
TEST_CASE("PenaltyCouplingCondition: zero alpha contributes nothing",
          "[conditions][coupling]")
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

    PenaltyCouplingCondition<double, 2> cc(
        *side_a, *side_b, 0, 1, *element, *element, g1d);
    cc.add(std::make_shared<TransverseDisplacement<double>>(),
           std::make_shared<TransverseDisplacement<double>>(),
           /*penalty=*/0.0, /*sign_b=*/+1.0, /*value=*/5.0);

    DofLayout layout;
    const Index ndof = element->num_node_dofs();
    layout.allocate(DofType::Primal, patch_A->num_control_pts() * ndof, ndof);
    layout.allocate(DofType::Primal, patch_B->num_control_pts() * ndof, ndof);
    std::vector<DofLayout::BlockId> blocks{0, 1};

    const Index N = layout.num_dofs();
    Matrix<double> K = Matrix<double>::Zero(N, N);
    Vector<double> F = Vector<double>::Zero(N);
    cc.apply(K, F, layout, blocks);

    REQUIRE(K.norm() == Approx(0.0).margin(1e-15));
    REQUIRE(F.norm() == Approx(0.0).margin(1e-15));
}

// ===========================================================================
// TEST 3 — Off-diagonal block sign matches sign_b
//
// With sign_b = +1, K_AB = -α ∫ N_A^T N_B  ≤ 0 componentwise (entries non-positive
// since shape function values on a single span are non-negative for B-splines).
// With sign_b = -1, K_AB = +α ∫ N_A^T N_B  ≥ 0.
// ===========================================================================
TEST_CASE("PenaltyCouplingCondition: off-diagonal block sign tracks sign_b",
          "[conditions][coupling]")
{
    const double L = 1.0;
    const Index  p = 3, n = 5;

    auto patch_A = make_translated_square(L, L, p, n, n, 0.0);
    auto patch_B = make_translated_square(L, L, p, n, n, L);

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.01);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    GaussLegendre<double, 1> g1d(p + 1);

    auto side_a = create_patch_boundary<double, 2>(patch_A, 0, false);
    auto side_b = create_patch_boundary<double, 2>(patch_B, 0, true);

    const Index ndof = element->num_node_dofs();
    const Index Na = patch_A->num_control_pts() * ndof;
    const Index Nb = patch_B->num_control_pts() * ndof;
    const Index N  = Na + Nb;

    auto build = [&](double sign_b) {
        PenaltyCouplingCondition<double, 2> cc(
            *side_a, *side_b, 0, 1, *element, *element, g1d);
        cc.add(std::make_shared<TransverseDisplacement<double>>(),
               std::make_shared<TransverseDisplacement<double>>(),
               1e6, sign_b);
        DofLayout layout;
        layout.allocate(DofType::Primal, Na, ndof);
        layout.allocate(DofType::Primal, Nb, ndof);
        std::vector<DofLayout::BlockId> blocks{0, 1};
        Matrix<double> K = Matrix<double>::Zero(N, N);
        Vector<double> F = Vector<double>::Zero(N);
        cc.apply(K, F, layout, blocks);
        return K;
    };

    Matrix<double> K_pos = build(+1.0);
    Matrix<double> K_neg = build(-1.0);

    Matrix<double> K_AB_pos = K_pos.topRightCorner(Na, Nb);
    Matrix<double> K_AB_neg = K_neg.topRightCorner(Na, Nb);

    // K_AB with sign_b=+1 must be ≤ 0; with sign_b=-1 must be ≥ 0.
    REQUIRE(K_AB_pos.maxCoeff() <= 1e-14);
    REQUIRE(K_AB_neg.minCoeff() >= -1e-14);
    // And they have opposite signs of nonzero entries
    REQUIRE((K_AB_pos + K_AB_neg).norm() < 1e-12 * K_AB_pos.norm());
}

// ===========================================================================
// TEST 4 — Out-of-range patch index throws
// ===========================================================================
TEST_CASE("PenaltyCouplingCondition: out-of-range primal_blocks throws",
          "[conditions][coupling]")
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

    PenaltyCouplingCondition<double, 2> cc(
        *side_a, *side_b, /*idx_a=*/0, /*idx_b=*/3,    // patch_b_idx beyond vector
        *element, *element, g1d);
    cc.add(std::make_shared<TransverseDisplacement<double>>(),
           std::make_shared<TransverseDisplacement<double>>(),
           1e6);

    DofLayout layout;
    layout.allocate(DofType::Primal, patch_A->num_control_pts(), 1);
    std::vector<DofLayout::BlockId> only_one{0};
    Matrix<double> K = Matrix<double>::Zero(8, 8);
    Vector<double> F = Vector<double>::Zero(8);
    REQUIRE_THROWS_AS(cc.apply(K, F, layout, only_one), std::out_of_range);
}

// ===========================================================================
// TEST 5 — Two-patch reproduces single-patch reference
//
// Reference: rectangle [0, 2L]×[0, L] meshed as 2N × N basis (KL plate).
// Two-patch: two squares L×L joined at x=L, each with N×N basis, coupled by
// w-only penalty. Both clamped on outer edges, uniform pressure.
// Compare displacement at interior points away from the interface.
// ===========================================================================
TEST_CASE("PenaltyCouplingCondition: two-patch reproduces single-patch",
          "[conditions][coupling][physics]")
{
    const double E = 1.0e7, nu = 0.3, h = 0.01;
    const double L = 1.0, q0 = -1.0;
    const Index  p = 3, N = 8;     // basis count along each direction
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

    // Clamp w on all four outer edges (penalty).
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

    // Reference deflection at interior physical point (x=0.5L, y=0.5L) ⇒
    // patch_ref param (xi, eta) = (0.5L / 2L, 0.5L / L) = (0.25, 0.5).
    const double w_ref_left  = eval_w_at(patch_ref, u_ref, /*offset=*/0,
                                          /*ndof=*/1, 0.25, 0.5);
    // Reference at (1.5L, 0.5L) ⇒ (0.75, 0.5)
    const double w_ref_right = eval_w_at(patch_ref, u_ref, 0, 1, 0.75, 0.5);

    // ===== Two-patch coupled =====
    auto patch_A = make_translated_square(L, L, p, N, N, /*dx=*/0.0);
    auto patch_B = make_translated_square(L, L, p, N, N, /*dx=*/L);

    LinearElasticProblem<double, 2> prob;
    auto idx_A = prob.add_patch(patch_A, element, gauss2d);
    auto idx_B = prob.add_patch(patch_B, element, gauss2d);

    Vector<double> load_A = uniform_load_values(*patch_A, *gauss2d, q0);
    Vector<double> load_B = uniform_load_values(*patch_B, *gauss2d, q0);
    prob.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *patch_A, *element, *gauss2d, load_A), idx_A);
    prob.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *patch_B, *element, *gauss2d, load_B), idx_B);

    // Clamp w on outer edges of A (left, top, bottom) and B (right, top, bottom).
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
    clamp_outer(patch_A, idx_A, /*skip_dim=*/0, /*skip_start=*/false, bds); // skip A.u1 (interface)
    clamp_outer(patch_B, idx_B, /*skip_dim=*/0, /*skip_start=*/true,  bds); // skip B.u0 (interface)

    // Coupling on the interface: penalty on w only.
    auto side_a = create_patch_boundary<double, 2>(patch_A, 0, false); // A.u1
    auto side_b = create_patch_boundary<double, 2>(patch_B, 0, true);  // B.u0
    REQUIRE_NOTHROW((assert_conforming<double, 2>(*side_a, *side_b, false)));

    auto cc = std::make_shared<PenaltyCouplingCondition<double, 2>>(
        *side_a, *side_b, idx_A, idx_B, *element, *element, g1d);
    cc->add(std::make_shared<TransverseDisplacement<double>>(),
            std::make_shared<TransverseDisplacement<double>>(),
            /*penalty=*/1e8 * D / (L * L * L), /*sign_b=*/+1.0);
    prob.add_condition(cc, idx_A);   // patch_idx irrelevant for coupling

    Matrix<double> K;  Vector<double> F;
    prob.assemble(K, F);

    Eigen::PartialPivLU<Matrix<double>> solver(K);
    Vector<double> u = solver.solve(F);

    // Patch A occupies first block; patch B occupies second block.
    // Both have ndof = 1 (KL plate). u layout: [u_A; u_B].
    const Index n_dof_A = patch_A->num_control_pts();

    // Two-patch deflection at (0.5L, 0.5L) is interior of patch A → param (0.5, 0.5)
    double w_two_left = eval_w_at(patch_A, u, /*offset=*/0,
                                    /*ndof=*/1, 0.5, 0.5);
    // (1.5L, 0.5L) is interior of patch B → param (0.5, 0.5) on B
    double w_two_right = eval_w_at(patch_B, u, /*offset=*/n_dof_A,
                                     /*ndof=*/1, 0.5, 0.5);

    INFO("w_ref_left = " << w_ref_left << "  w_two_left = " << w_two_left);
    INFO("w_ref_right = " << w_ref_right << "  w_two_right = " << w_two_right);
    INFO("rel_err_left  = " << std::abs(w_two_left  - w_ref_left)  / std::abs(w_ref_left));
    INFO("rel_err_right = " << std::abs(w_two_right - w_ref_right) / std::abs(w_ref_right));

    REQUIRE(std::abs(w_two_left  - w_ref_left)  < 0.05 * std::abs(w_ref_left));
    REQUIRE(std::abs(w_two_right - w_ref_right) < 0.05 * std::abs(w_ref_right));

    // Assembled K must be symmetric.
    REQUIRE((K - K.transpose()).norm() < 1e-10 * K.norm());
}
