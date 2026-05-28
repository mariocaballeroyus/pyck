#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "boundary_field.hpp"
#include "patch_boundary.hpp"
#include "gauss_legendre.hpp"
#include "quadrature.hpp"
#include "linear_elastic_problem.hpp"
#include "load_boundary_condition.hpp"
#include "boundary_lagrange_condition.hpp"
#include "dof_layout.hpp"
#include "plane_stress_2d.hpp"
#include "plate_reissner_mindlin_3p.hpp"
#include <Eigen/Dense>

using namespace pyck;

// ===========================================================================
// TEST 1 — Structural sanity:
//   K is untouched and F integrates to (Q × edge_length) on the w slot.
// ===========================================================================
TEST_CASE("LoadBoundaryCondition: K is unchanged and F sums to Q*L_edge",
          "[conditions][load_boundary]")
{
    Index p = 2, n = 5;
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    const double L = 1.0, W = 1.0;
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, L, W));

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.1);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    GaussLegendre<double, 1> g1(p + 1);

    // Right edge (dim 0, at_start = false): outward normal = (+1, 0), length = W.
    auto bdy = create_patch_boundary<double, 2>(surface, 0, false);

    const double Q = 5.0;
    LoadBoundaryCondition<double, 2> cond(*bdy, *element, g1);
    cond.add(std::make_shared<TransverseDisplacement<double>>(), Q);

    DofLayout layout;
    auto primal = layout.allocate(
        DofType::Primal, surface->num_control_pts() * 3, 3);

    const Index N = surface->num_control_pts() * 3;
    Matrix<double> K = Matrix<double>::Zero(N, N);
    Vector<double> F = Vector<double>::Zero(N);
    cond.apply(K, F, layout, primal);

    REQUIRE(K.norm() == Approx(0.0).margin(1e-15));

    double f_w_sum = 0.0;
    for (Index i = 0; i < surface->num_control_pts(); ++i) {
        f_w_sum += F(3 * i + 0);   // slot 0 = w
    }
    REQUIRE(f_w_sum == Approx(Q * W).margin(1e-12));

    // Slots 1 and 2 (rotations) must remain at zero — TransverseDisplacement
    // only touches the w column of N_v.
    double f_rot_sum = 0.0;
    for (Index i = 0; i < surface->num_control_pts(); ++i) {
        f_rot_sum += std::abs(F(3 * i + 1)) + std::abs(F(3 * i + 2));
    }
    REQUIRE(f_rot_sum == Approx(0.0).margin(1e-14));
}

// ===========================================================================
// TEST 2 — Physics:
//   RM-3p plate strip with FREE lateral edges (y = 0, y = W). Clamped at
//   x = 0; distributed shear Q on the x = L edge. Because the lateral edges
//   are free of moment, the strip behaves as an Euler-Bernoulli beam with
//   EI = E W h^3 / 12 (no 1 - nu^2 factor), giving
//        w_tip = (Q W) L^3 / (3 E I) = 4 Q L^3 / (E h^3).
//   Tolerance 2 % at p = 3, n = 12 (IGA discretisation slack).
// ===========================================================================
TEST_CASE("LoadBoundaryCondition: cantilever tip deflection matches closed form",
          "[conditions][load_boundary]")
{
    const double E  = 1.0e7;
    const double nu = 0.3;
    const double h  = 0.1;
    const double L  = 10.0;        // x-direction
    const double W  = 1.0;         // y-direction (per-unit-width strip)
    const double Q  = 1.0;         // shear per unit edge length on right edge

    Index p = 3, n_elem = 12;
    int   nq = p + 1;

    auto bsp_u = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    auto bsp_v = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, 4));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp_u, bsp_v, L, W));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(nq);
    GaussLegendre<double, 1> gauss1d(nq);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);

    // Clamp left edge (dim 0, at_start = true): w = phi_n = phi_s = 0.
    auto left_bdy = create_patch_boundary<double, 2>(surface, 0, true);
    auto clamp = std::make_shared<LagrangeBoundaryCondition<double, 2>>(
        *left_bdy, *element, gauss1d);
    clamp->add(std::make_shared<TransverseDisplacement<double>>(), 0.0);
    clamp->add(std::make_shared<NormalRotation<double>>(),         0.0);
    clamp->add(std::make_shared<TangentialRotation<double>>(),     0.0);
    problem.add_condition(clamp);

    // Apply distributed shear Q on right edge.
    auto right_bdy = create_patch_boundary<double, 2>(surface, 0, false);
    auto neumann = std::make_shared<LoadBoundaryCondition<double, 2>>(
        *right_bdy, *element, gauss1d);
    neumann->add(std::make_shared<TransverseDisplacement<double>>(), Q);
    problem.add_condition(neumann);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    Eigen::PartialPivLU<Matrix<double>> solver(K);
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    // Evaluate w at the tip-centerline (xi = 1, eta = 0.5).
    ColMatrix<double, 2> pt(1, 2);
    pt << 1.0, 0.5;
    Index su = bsp_u->knot_vector().find_span(p, 1.0);
    Index sv = bsp_v->knot_vector().find_span(p, 0.5);
    Index ni_u = surface->tensor_product().num_intervals()[0];
    Index flat = su + sv * ni_u;

    const auto b = (*surface).tensor_product().eval_all(pt, 0);
    std::vector<Index> active;
    surface->dof_mapper().get_element_cps(flat, active);
    Vector<double> w_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i) {
        w_active(i) = u(active[i] * 3 + 0);
    }
    const double w_num = b[0].col(0).dot(w_active);

    // Free-edge plate strip cantilever: EI = E W h^3 / 12.
    //   total tip force = Q * W, w_tip = (Q W) L^3 / (3 E I) = 4 Q L^3 / (E h^3).
    const double w_exact = 4.0 * Q * L*L*L / (E * h*h*h);

    INFO("w_num = " << w_num << "  w_exact = " << w_exact);
    REQUIRE(std::abs(w_num - w_exact) / std::abs(w_exact) < 2e-2);
}

// ===========================================================================
// TEST 3 — Per-qpt array overload:
//   Computes a constant Q traction first via the scalar overload, then via
//   the per-qpt array overload (filled with the same constant), and checks
//   that the assembled F vectors agree to floating-point precision.
// ===========================================================================
TEST_CASE("LoadBoundaryCondition: scalar and per-qpt array overloads agree",
          "[conditions][load_boundary]")
{
    Index p = 2, n = 5;
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, 1.0, 1.0));

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.1);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    GaussLegendre<double, 1> g1(p + 1);

    auto bdy = create_patch_boundary<double, 2>(surface, 1, false);  // top edge
    const double Q = 3.7;

    DofLayout layout;
    auto primal = layout.allocate(
        DofType::Primal, surface->num_control_pts() * 3, 3);
    const Index N = surface->num_control_pts() * 3;

    // Scalar path.
    Vector<double> F_scalar = Vector<double>::Zero(N);
    {
        LoadBoundaryCondition<double, 2> cond(*bdy, *element, g1);
        cond.add(std::make_shared<TransverseDisplacement<double>>(), Q);
        Matrix<double> K = Matrix<double>::Zero(N, N);
        cond.apply(K, F_scalar, layout, primal);
    }

    // Per-qpt array path: same constant value at every active qpt.
    Vector<double> F_array = Vector<double>::Zero(N);
    {
        LoadBoundaryCondition<double, 2> cond(*bdy, *element, g1);
        const Index nq = cond.num_active_qpts();
        Vector<double> values = Vector<double>::Constant(nq, Q);
        cond.add(std::make_shared<TransverseDisplacement<double>>(), values);
        Matrix<double> K = Matrix<double>::Zero(N, N);
        cond.apply(K, F_array, layout, primal);
    }

    REQUIRE((F_scalar - F_array).norm() < 1e-14);
}
