#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "patch.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "boundary_field.hpp"
#include "patch_boundary.hpp"
#include "gauss_legendre.hpp"
#include "quadrature.hpp"
#include "linear_elastic_problem.hpp"
#include "boundary_penalty_condition.hpp"
#include "dof_layout.hpp"
#include "plane_stress_2d.hpp"
#include "plate_kirchhoff_love_1p.hpp"

using namespace pyck;

// ---------------------------------------------------------------------------
// Build a tagged patch: square KL plate of side L, p=p, n_elem×n_elem grid.
// Returns owned objects via shared_ptr; element + quadrature are returned by
// reference parameter so they can be reused across problems.
// ---------------------------------------------------------------------------
static Ptr<Patch<double, 2>> make_square_plate(
    double L,
    Index p,
    Index n_elem)
{
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    return std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, L, L));
}

// ---------------------------------------------------------------------------
// Apply a w-only penalty on all four edges of a patch.
// ---------------------------------------------------------------------------
static void clamp_w_all_edges(
    LinearElasticProblem<double, 2>& problem,
    std::vector<Ptr<PatchBoundary<double, 2>>>& boundaries,
    const Ptr<Patch<double, 2>>& surface,
    const Element<double, 2>& element,
    const QuadratureRule<double, 1>& gauss1d,
    double alpha)
{
    for (std::size_t dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto bdy = create_patch_boundary<double, 2>(surface, dim, start);
            boundaries.push_back(bdy);
            auto cond = std::make_shared<PenaltyBoundaryCondition<double, 2>>(
                *boundaries.back(), element, gauss1d);
            cond->add(std::make_shared<TransverseDisplacement<double>>(),
                      alpha, 0.0);
            problem.add_condition(cond);
        }
    }
}

// ===========================================================================
// TEST 1 — Single-patch vector-constructor regression
// ---------------------------------------------------------------------------
// Assembling via the legacy single-patch ctor and via the new vector ctor
// must produce identical K and F.
// ===========================================================================
TEST_CASE("LinearElasticProblem single-patch vector ctor matches legacy ctor",
          "[multipatch][regression]")
{
    const double E = 1.0e7, nu = 0.3, h = 0.01;
    const double L = 1.0, q0 = -1.0;
    const Index  p = 3, n_elem = 6;

    auto surface  = make_square_plate(L, p, n_elem);
    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);
    GaussLegendre<double, 1> gauss1d(p + 1);

    const double D = E * h*h*h / (12.0 * (1.0 - nu*nu));
    const double alpha = 1e6 * D / (L * L * L);

    // -------- Legacy single-patch ----------------------------------------
    LinearElasticProblem<double, 2> legacy(surface, element, gauss2d);

    legacy.add_domain_load(*surface, (Vector<double>(3) << 0.0, 0.0, q0).finished());

    std::vector<Ptr<PatchBoundary<double, 2>>> bds_legacy;
    clamp_w_all_edges(legacy, bds_legacy, surface, *element, gauss1d, alpha);

    SparseMatrix<double> K_legacy_sparse;
    Vector<double> F_legacy;
    legacy.assemble(K_legacy_sparse, F_legacy);
    Matrix<double> K_legacy = Matrix<double>(K_legacy_sparse);

    // -------- New vector ctor (single patch) -----------------------------
    LinearElasticProblem<double, 2> vec_problem(
        std::vector<Ptr<Patch<double, 2>>>{surface},
        std::vector<Ptr<Element<double, 2>>>{element},
        std::vector<Ptr<QuadratureRule<double, 2>>>{gauss2d});

    REQUIRE(vec_problem.num_patches() == 1);

    vec_problem.add_domain_load(*surface, (Vector<double>(3) << 0.0, 0.0, q0).finished());

    std::vector<Ptr<PatchBoundary<double, 2>>> bds_vec;
    clamp_w_all_edges(vec_problem, bds_vec, surface, *element, gauss1d, alpha);

    SparseMatrix<double> K_vec_sparse;
    Vector<double> F_vec;
    vec_problem.assemble(K_vec_sparse, F_vec);
    Matrix<double> K_vec = Matrix<double>(K_vec_sparse);

    // -------- Assertions --------------------------------------------------
    REQUIRE(K_vec.rows() == K_legacy.rows());
    REQUIRE(K_vec.cols() == K_legacy.cols());
    REQUIRE(F_vec.size() == F_legacy.size());
    REQUIRE((K_vec - K_legacy).norm() < 1e-12 * K_legacy.norm());
    REQUIRE((F_vec - F_legacy).norm() < 1e-12 * F_legacy.norm());
}

// ===========================================================================
// TEST 2 — Two disconnected patches assemble to a block-diagonal system
// ---------------------------------------------------------------------------
// Solving {patch_A, patch_B} together must yield the same per-patch K and F
// as solving them independently. K must have zero off-diagonal patch blocks.
// ===========================================================================
TEST_CASE("LinearElasticProblem two disconnected patches → block-diagonal",
          "[multipatch][block_diagonal]")
{
    const double E = 1.0e7, nu = 0.3, h = 0.01;
    const Index  p = 3;

    // Distinct geometries so the two diagonal blocks are not identical.
    const double L_A = 1.0, q_A = -1.0;
    const Index  n_A = 5;
    const double L_B = 0.7, q_B = -2.5;
    const Index  n_B = 4;

    auto surf_A = make_square_plate(L_A, p, n_A);
    auto surf_B = make_square_plate(L_B, p, n_B);

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);
    GaussLegendre<double, 1> gauss1d(p + 1);

    const double D = E * h*h*h / (12.0 * (1.0 - nu*nu));

    const double alpha_A = 1e6 * D / (L_A * L_A * L_A);
    const double alpha_B = 1e6 * D / (L_B * L_B * L_B);

    // -------- Standalone reference solves --------------------------------
    LinearElasticProblem<double, 2> ref_A(surf_A, element, gauss2d);
    ref_A.add_domain_load(*surf_A, (Vector<double>(3) << 0.0, 0.0, q_A).finished());
    std::vector<Ptr<PatchBoundary<double, 2>>> bd_A_ref;
    clamp_w_all_edges(ref_A, bd_A_ref, surf_A, *element, gauss1d, alpha_A);
    SparseMatrix<double> K_A_ref_sparse;
    Vector<double> F_A_ref;
    ref_A.assemble(K_A_ref_sparse, F_A_ref);
    Matrix<double> K_A_ref = Matrix<double>(K_A_ref_sparse);

    LinearElasticProblem<double, 2> ref_B(surf_B, element, gauss2d);
    ref_B.add_domain_load(*surf_B, (Vector<double>(3) << 0.0, 0.0, q_B).finished());
    std::vector<Ptr<PatchBoundary<double, 2>>> bd_B_ref;
    clamp_w_all_edges(ref_B, bd_B_ref, surf_B, *element, gauss1d, alpha_B);
    SparseMatrix<double> K_B_ref_sparse;
    Vector<double> F_B_ref;
    ref_B.assemble(K_B_ref_sparse, F_B_ref);
    Matrix<double> K_B_ref = Matrix<double>(K_B_ref_sparse);

    // -------- Combined two-patch problem --------------------------------
    LinearElasticProblem<double, 2> combined;
    const std::size_t idx_A = combined.add_patch(surf_A, element, gauss2d);
    const std::size_t idx_B = combined.add_patch(surf_B, element, gauss2d);
    REQUIRE(idx_A == 0);
    REQUIRE(idx_B == 1);
    REQUIRE(combined.num_patches() == 2);

    combined.add_domain_load(*surf_A, (Vector<double>(3) << 0.0, 0.0, q_A).finished());
    combined.add_domain_load(*surf_B, (Vector<double>(3) << 0.0, 0.0, q_B).finished());

    std::vector<Ptr<PatchBoundary<double, 2>>> bd_combined;
    clamp_w_all_edges(combined, bd_combined, surf_A, *element, gauss1d, alpha_A);
    clamp_w_all_edges(combined, bd_combined, surf_B, *element, gauss1d, alpha_B);

    SparseMatrix<double> K_AB_sparse;
    Vector<double> F_AB;
    combined.assemble(K_AB_sparse, F_AB);
    Matrix<double> K_AB = Matrix<double>(K_AB_sparse);

    // -------- Sizes -----------------------------------------------------
    const Index n_dof_A = K_A_ref.rows();
    const Index n_dof_B = K_B_ref.rows();
    REQUIRE(K_AB.rows() == n_dof_A + n_dof_B);
    REQUIRE(K_AB.cols() == n_dof_A + n_dof_B);
    REQUIRE(F_AB.size() == n_dof_A + n_dof_B);

    // -------- Block decomposition ---------------------------------------
    Matrix<double> K_AA = K_AB.topLeftCorner(n_dof_A, n_dof_A);
    Matrix<double> K_BB = K_AB.bottomRightCorner(n_dof_B, n_dof_B);
    Matrix<double> K_offAB = K_AB.topRightCorner(n_dof_A, n_dof_B);
    Matrix<double> K_offBA = K_AB.bottomLeftCorner(n_dof_B, n_dof_A);

    Vector<double> F_A_blk = F_AB.head(n_dof_A);
    Vector<double> F_B_blk = F_AB.tail(n_dof_B);

    REQUIRE((K_AA - K_A_ref).norm() < 1e-12 * K_A_ref.norm());
    REQUIRE((K_BB - K_B_ref).norm() < 1e-12 * K_B_ref.norm());
    REQUIRE((F_A_blk - F_A_ref).norm() < 1e-12 * F_A_ref.norm());
    REQUIRE((F_B_blk - F_B_ref).norm() < 1e-12 * F_B_ref.norm());

    // Off-diagonal patch blocks must be exactly zero (no coupling).
    REQUIRE(K_offAB.norm() < 1e-14);
    REQUIRE(K_offBA.norm() < 1e-14);

    // -------- Combined solve matches independent solves -----------------
    Eigen::PartialPivLU<Matrix<double>> solver(K_AB);
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u_AB = solver.solve(F_AB);

    Eigen::PartialPivLU<Matrix<double>> solver_A(K_A_ref);
    Vector<double> u_A_ref = solver_A.solve(F_A_ref);
    Eigen::PartialPivLU<Matrix<double>> solver_B(K_B_ref);
    Vector<double> u_B_ref = solver_B.solve(F_B_ref);

    Vector<double> u_A_blk = u_AB.head(n_dof_A);
    Vector<double> u_B_blk = u_AB.tail(n_dof_B);

    REQUIRE((u_A_blk - u_A_ref).norm() < 1e-10 * std::max(u_A_ref.norm(), 1e-30));
    REQUIRE((u_B_blk - u_B_ref).norm() < 1e-10 * std::max(u_B_ref.norm(), 1e-30));
}

// ===========================================================================
// TEST 3 — Builder API: empty ctor + add_patch + per-patch add_condition
// ===========================================================================
TEST_CASE("LinearElasticProblem builder API: empty ctor and add_patch",
          "[multipatch][builder]")
{
    const double E = 1.0e7, nu = 0.3, h = 0.01;
    const Index  p = 2, n_elem = 3;
    const double L = 1.0;

    auto surf = make_square_plate(L, p, n_elem);
    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);

    LinearElasticProblem<double, 2> problem;
    REQUIRE(problem.num_patches() == 0);

    // assembling with no patches must throw a clear error
    {
        SparseMatrix<double> K;
        Vector<double> F;
        REQUIRE_THROWS_AS(problem.assemble(K, F), std::runtime_error);
    }

    const std::size_t idx = problem.add_patch(surf, element, gauss2d);
    REQUIRE(idx == 0);
    REQUIRE(problem.num_patches() == 1);

    // A condition whose patch is not registered with the problem must throw.
    // Build it on a separate, unregistered patch.
    auto surf_other = make_square_plate(L, p, n_elem);
    REQUIRE_THROWS_AS(problem.add_domain_load(*surf_other, (Vector<double>(3) << 0.0, 0.0, 0.0).finished()),
                      std::invalid_argument);
}
