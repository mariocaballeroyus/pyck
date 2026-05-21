#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "physical_points.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "boundary_field.hpp"
#include "patch_boundary.hpp"
#include "gauss_legendre.hpp"
#include "quadrature.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "direct_constraint.hpp"
#include "penalty_boundary_condition.hpp"
#include "dof_layout.hpp"
#include "plane_stress_2d.hpp"
#include "plate_kirchhoff_love_1p.hpp"
#include "plate_reissner_mindlin_3p.hpp"
#include "plate_reissner_mindlin_1p.hpp"
#include <Eigen/Dense>

using namespace pyck;

// ---------------------------------------------------------------------------
// Navier bi-sinusoidal solution (KL / thin plate) at a single point
// ---------------------------------------------------------------------------
static double navier_bisin(double x, double y, double L, double W,
                            double f0, double D)
{
    const double pi = M_PI;
    return (f0 / (D * std::pow(pi*pi*(1.0/(L*L) + 1.0/(W*W)), 2)))
           * std::sin(pi * x / L) * std::sin(pi * y / W);
}

// ---------------------------------------------------------------------------
// Navier series for uniform load on a SS square plate (thin-plate limit)
// ---------------------------------------------------------------------------
static double navier_uniform(double x, double y, double a,
                              double q0, double D, int N = 100)
{
    const double pi = M_PI;
    double w = 0.0;
    for (int m = 1; m <= N; m += 2)
        for (int n = 1; n <= N; n += 2) {
            double denom = m * n * std::pow((double)(m*m + n*n), 2.0);
            w += std::sin(m * pi * x / a) * std::sin(n * pi * y / a) / denom;
        }
    return 16.0 * q0 / (std::pow(pi, 6.0) * D) * w;
}

// ---------------------------------------------------------------------------
// Helper: apply penalty BCs on all four edges of a 2D patch
// ---------------------------------------------------------------------------
template <typename T>
static void add_penalty_all_edges(
    LinearElasticProblem<T, 2>& problem,
    std::vector<Ptr<PatchBoundary<T, 2>>>& boundaries,
    const Ptr<Patch<T, 2>>& surface,
    const Element<T, 2>&    element,
    const QuadratureRule<T, 1>& gauss1d,
    T penalty_w,    T value_w,
    T penalty_rot_n = T(0), T value_rot_n = T(0),
    T penalty_rot_s = T(0), T value_rot_s = T(0))
{
    for (std::size_t dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto bdy = create_patch_boundary<T, 2>(surface, dim, start);
            boundaries.push_back(bdy);
            auto cond = std::make_shared<PenaltyBoundaryCondition<T, 2>>(
                *boundaries.back(), element, gauss1d);
            if (penalty_w != T(0)) {
                cond->add(std::make_shared<TransverseDisplacement<T>>(),
                          penalty_w, value_w);
            }
            if (penalty_rot_n != T(0)) {
                cond->add(std::make_shared<NormalRotation<T>>(),
                          penalty_rot_n, value_rot_n);
            }
            if (penalty_rot_s != T(0)) {
                cond->add(std::make_shared<TangentialRotation<T>>(),
                          penalty_rot_s, value_rot_s);
            }
            problem.add_condition(cond);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: evaluate w at centre (ξ=0.5, η=0.5) from a displacement vector
// DOF layout: (w=0, ...) per node, ndof DOFs per node
// ---------------------------------------------------------------------------
static double eval_w_centre(const Ptr<Patch<double,2>>& surf,
                             const Vector<double>& u,
                             Index ndof)
{
    auto bsp_u = surf->basis_ptr(0);
    auto bsp_v = surf->basis_ptr(1);
    ColMatrix<double, 2> pt(1, 2);
    pt << 0.5, 0.5;
    Index su = bsp_u->find_span(0.5);
    Index sv = bsp_v->find_span(0.5);
    Index num_su = surf->tensor_product().num_intervals()[0];
    Index flat   = su + sv * num_su;   // u-fastest

    const auto b = (*surf).tensor_product().eval_all(pt, 0);
    std::vector<Index> active;
    surf->dof_mapper().get_element_dofs(flat, active);

    Vector<double> w_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        w_active(i) = u(active[i] * ndof + 0);
    return b[0].col(0).dot(w_active);
}

// ===========================================================================
// TEST 1 — Structural checks: symmetry and zero-alpha no-contribution
// ===========================================================================
TEST_CASE("PenaltyBoundaryCondition structural properties", "[conditions][penalty]")
{
    Index p = 2, n = 4;
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, 1.0, 1.0));

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.01);
    auto elem_rm3 = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    GaussLegendre<double, 1> g1(p + 1);

    auto bdy = create_patch_boundary<double, 2>(surface, 0, false);  // right edge

    // Total DOFs = n_nodes_2d * ndof_per_node = (n*n) * 3
    const Index N_dof_rm3 = surface->num_control_pts() * 3;

    auto setup_layout_rm3 = [&]() {
        DofLayout layout;
        auto primal = layout.allocate(pyck::DofType::Primal, surface->num_control_pts() * 3, 3);
        return std::make_pair(std::move(layout), primal);
    };

    SECTION("Stiffness is symmetric for w-only penalty")
    {
        PenaltyBoundaryCondition<double, 2> pen(*bdy, *elem_rm3, g1);
        pen.add(std::make_shared<TransverseDisplacement<double>>(), 1e6, 0.0);
        auto [layout, primal] = setup_layout_rm3();
        Matrix<double> K = Matrix<double>::Zero(N_dof_rm3, N_dof_rm3);
        Vector<double> F = Vector<double>::Zero(N_dof_rm3);
        pen.apply(K, F, layout, std::vector<DofLayout::BlockId>{primal});
        REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());
    }

    SECTION("Stiffness is symmetric with all three penalties")
    {
        PenaltyBoundaryCondition<double, 2> pen(*bdy, *elem_rm3, g1);
        pen.add(std::make_shared<TransverseDisplacement<double>>(), 1e6, 0.0);
        pen.add(std::make_shared<NormalRotation<double>>(), 1e6, 0.0);
        pen.add(std::make_shared<TangentialRotation<double>>(), 1e6, 0.0);
        auto [layout, primal] = setup_layout_rm3();
        Matrix<double> K = Matrix<double>::Zero(N_dof_rm3, N_dof_rm3);
        Vector<double> F = Vector<double>::Zero(N_dof_rm3);
        pen.apply(K, F, layout, std::vector<DofLayout::BlockId>{primal});
        REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());
    }

    SECTION("Zero alpha contributes nothing to K and F")
    {
        PenaltyBoundaryCondition<double, 2> pen(*bdy, *elem_rm3, g1);
        pen.add(std::make_shared<TransverseDisplacement<double>>(), 0.0, 5.0);
        pen.add(std::make_shared<NormalRotation<double>>(), 0.0, 3.0);
        pen.add(std::make_shared<TangentialRotation<double>>(), 0.0, 1.0);
        auto [layout, primal] = setup_layout_rm3();
        Index N_dof = surface->num_control_pts() * 3;
        Matrix<double> K = Matrix<double>::Zero(N_dof, N_dof);
        Vector<double> F = Vector<double>::Zero(N_dof);
        pen.apply(K, F, layout, std::vector<DofLayout::BlockId>{primal});
        REQUIRE(K.norm() == Approx(0.0).margin(1e-15));
        REQUIRE(F.norm() == Approx(0.0).margin(1e-15));
    }

    SECTION("KL element (1 DOF): penalty K is symmetric and positive semi-definite")
    {
        auto elem_kl = std::make_shared<PlateKirchhoffLove1p<double>>(material);
        auto bdy_kl  = create_patch_boundary<double, 2>(surface, 1, true);  // bottom edge
        PenaltyBoundaryCondition<double, 2> pen(*bdy_kl, *elem_kl, g1);
        pen.add(std::make_shared<TransverseDisplacement<double>>(), 1e6, 0.0);

        DofLayout layout;
        auto primal = layout.allocate(pyck::DofType::Primal, surface->num_control_pts() * 1, 1);

        Index N_dof = surface->num_control_pts();
        Matrix<double> K = Matrix<double>::Zero(N_dof, N_dof);
        Vector<double> F = Vector<double>::Zero(N_dof);
        pen.apply(K, F, layout, std::vector<DofLayout::BlockId>{primal});

        REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());
        // All eigenvalues >= 0 (PSD)
        Eigen::SelfAdjointEigenSolver<Matrix<double>> es(K);
        REQUIRE(es.eigenvalues().minCoeff() >= -1e-10);
    }
}

// ===========================================================================
// TEST 2 — KL plate: penalty SS BCs reproduce Navier bi-sinusoidal solution
// ===========================================================================
TEST_CASE("PenaltyBoundaryCondition KL plate: SS via penalty matches Navier", "[conditions][penalty]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double L  = 2.0, W = 1.0;
    double f0 = -10.0;

    Index p = 3, n_elem = 10;
    int   nq = p + 1;
    double D = E * h*h*h / (12.0 * (1.0 - nu*nu));

    auto bsp_u = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    auto bsp_v = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp_u, bsp_v, L, W));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element  = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(nq);
    GaussLegendre<double, 1> gauss1d(nq);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);

    // Variable load q(x,y) = f0 * sin(πx/L) sin(πy/W)
    auto phys_pts = eval_physical_points<double, 2>(*surface, *gauss2d);
    Vector<double> load_vals(phys_pts.rows());
    for (Index i = 0; i < phys_pts.rows(); ++i)
        load_vals(i) = f0 * std::sin(M_PI * phys_pts(i,0) / L)
                          * std::sin(M_PI * phys_pts(i,1) / W);

    problem.add_condition(std::make_shared<LoadCondition<double,2>>(
        *surface, *element, *gauss2d, load_vals));

    // Penalty factor: α >> D/L so BCs are accurately enforced.
    // For penalty BCs, a relative error O(D/(α*L³)) compared to the exact BC
    // is expected; 1e8 keeps this below 0.5 %.
    const double alpha = 1e8 * D / (L * L * L);
    std::vector<Ptr<PatchBoundary<double, 2>>> penalty_boundaries;
    add_penalty_all_edges<double>(
        problem, penalty_boundaries, surface, *element, gauss1d, alpha, 0.0);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    Eigen::PartialPivLU<Matrix<double>> solver(K);
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    // Verify deflection at a grid of interior points
    std::vector<double> test_coords = {0.25, 0.5, 0.75};
    for (double pu : test_coords) {
        for (double pv : test_coords) {
            ColMatrix<double, 2> pt(1, 2);
            pt << pu, pv;
            Index su = bsp_u->knot_vector().find_span(p, pu);
            Index sv = bsp_v->knot_vector().find_span(p, pv);
            Index ni_u = surface->tensor_product().num_intervals()[0];
            Index flat = su + sv * ni_u;

            const auto b = (*surface).tensor_product().eval_all(pt, 0);
            std::vector<Index> active;
            surface->dof_mapper().get_element_dofs(flat, active);
            Vector<double> u_a(active.size());
            for (std::size_t i = 0; i < active.size(); ++i)
                u_a(i) = u(active[i]);

            double w_num   = b[0].col(0).dot(u_a);
            double w_exact = navier_bisin(pu * L, pv * W, L, W, f0, D);
            double rel_err = std::abs(w_num - w_exact) / std::abs(w_exact);

            INFO("(" << pu*L << ", " << pv*W << ")  w_exact=" << w_exact
                 << "  w_num=" << w_num);
            REQUIRE(rel_err < 5e-3);
        }
    }
}

// ===========================================================================
// TEST 3 — RM-1p plate: penalty SS BCs reproduce Navier solution
// ===========================================================================
TEST_CASE("PenaltyBoundaryCondition RM-1p plate: SS via penalty matches Navier", "[conditions][penalty]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;  // thin: RM-1p ≈ KL
    double a  = 1.0;
    double q0 = -1.0;
    double D  = E * h*h*h / (12.0 * (1.0 - nu*nu));

    Index p = 3, n_elem = 8;
    int   nq = p + 1;

    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    double k_shear = 5.0 / 6.0;
    auto material  = std::make_shared<PlaneStress2d<double>>(E, nu, h, k_shear);
    auto element   = std::make_shared<PlateReissnerMindlin1p<double>>(material);
    auto gauss2d   = std::make_shared<GaussLegendre<double, 2>>(nq);
    GaussLegendre<double, 1> gauss1d(nq);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);

    // Uniform load (full DOF vector)
    Index n_spans = bsp->knot_vector().num_spans();
    Index active_elems = 0;
    for (Index e = 0; e < n_spans * n_spans; ++e) {
        Index su = e % n_spans;
        Index sv = e / n_spans;
        auto [lou, hiu] = bsp->knot_vector().span_bounds(su);
        auto [lov, hiv] = bsp->knot_vector().span_bounds(sv);
        if (std::abs(hiu-lou) > 1e-14 && std::abs(hiv-lov) > 1e-14)
            ++active_elems;
    }
    const Index Q = gauss2d->num_points();
    Vector<double> load_vals = Vector<double>::Constant(active_elems * Q, q0);

    problem.add_condition(std::make_shared<LoadCondition<double,2>>(
        *surface, *element, *gauss2d, load_vals));

    const double alpha = 1e6 * D / (a * a * a);
    std::vector<Ptr<PatchBoundary<double, 2>>> penalty_boundaries;
    add_penalty_all_edges<double>(
        problem, penalty_boundaries, surface, *element, gauss1d, alpha, 0.0);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K.sparseView());
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    double w_num    = eval_w_centre(surface, u, 1);
    double w_exact  = navier_uniform(a/2, a/2, a, q0, D, 200);
    double rel_err  = std::abs(w_num - w_exact) / std::abs(w_exact);

    INFO("RM-1p SS: w_exact=" << w_exact << "  w_num=" << w_num
         << "  rel_err=" << rel_err);
    REQUIRE(rel_err < 3e-2);
}

// ===========================================================================
// TEST 4 — RM-3p plate: w-only penalty (SS) matches Navier
// ===========================================================================
TEST_CASE("PenaltyBoundaryCondition RM-3p plate: SS (w-only) matches Navier", "[conditions][penalty]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    double q0 = -1.0;
    double D  = E * h*h*h / (12.0 * (1.0 - nu*nu));

    Index p = 3, n_elem = 8;
    int   nq = p + 1;

    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    double k_shear = 5.0 / 6.0;
    auto material  = std::make_shared<PlaneStress2d<double>>(E, nu, h, k_shear);
    auto element   = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss2d   = std::make_shared<GaussLegendre<double, 2>>(nq);
    GaussLegendre<double, 1> gauss1d(nq);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);

    // Uniform load (only w-DOF = 0 carries the load)
    Index n_spans = bsp->knot_vector().num_spans();
    Index active_elems = 0;
    for (Index e = 0; e < n_spans * n_spans; ++e) {
        Index su = e % n_spans;
        Index sv = e / n_spans;
        auto [lou, hiu] = bsp->knot_vector().span_bounds(su);
        auto [lov, hiv] = bsp->knot_vector().span_bounds(sv);
        if (std::abs(hiu-lou) > 1e-14 && std::abs(hiv-lov) > 1e-14)
            ++active_elems;
    }
    const Index Q    = gauss2d->num_points();
    const Index ndof = 3;
    Vector<double> load_vals = Vector<double>::Zero(active_elems * Q * ndof);
    for (Index i = 0; i < active_elems * Q; ++i)
        load_vals(i * ndof + 0) = q0;

    problem.add_condition(std::make_shared<LoadCondition<double,2>>(
        *surface, *element, *gauss2d, load_vals));

    // Only penalise w; rotations are free (SS condition)
    const double penalty_w = 1e6 * D / (a * a * a);
    std::vector<Ptr<PatchBoundary<double, 2>>> penalty_boundaries;
    add_penalty_all_edges<double>(problem, penalty_boundaries, surface, *element, gauss1d,
                                  penalty_w, 0.0);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K.sparseView());
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    double w_num   = eval_w_centre(surface, u, ndof);
    double w_exact = navier_uniform(a/2, a/2, a, q0, D, 200);
    double rel_err = std::abs(w_num - w_exact) / std::abs(w_exact);

    INFO("RM-3p SS (w-penalty): w_exact=" << w_exact
         << "  w_num=" << w_num << "  rel_err=" << rel_err);
    REQUIRE(rel_err < 3e-2);
}

// ===========================================================================
// TEST 5 — RM-3p plate: clamped via penalty agrees with DirectConstraint reference
// ===========================================================================
TEST_CASE("PenaltyBoundaryCondition RM-3p plate: clamped BCs agree with DirectConstraint",
          "[conditions][penalty]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    double q0 = -1.0;
    double D  = E * h*h*h / (12.0 * (1.0 - nu*nu));

    Index p = 3, n_elem = 8;
    int   nq = p + 1;

    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));

    auto make_surface = [&]() {
        return std::make_shared<Patch<double, 2>>(
            rectangle<double>(bsp, bsp, a, a));
    };

    double k_shear = 5.0 / 6.0;
    auto material  = std::make_shared<PlaneStress2d<double>>(E, nu, h, k_shear);
    auto element   = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    const Index ndof = element->num_node_dofs();  // 3
    auto gauss2d   = std::make_shared<GaussLegendre<double, 2>>(nq);
    GaussLegendre<double, 1> gauss1d(nq);

    // Build load values once
    auto surface_ref = make_surface();
    Index n_spans = bsp->knot_vector().num_spans();
    Index active_elems = 0;
    for (Index e = 0; e < n_spans * n_spans; ++e) {
        Index su = e % n_spans;
        Index sv = e / n_spans;
        auto [lou, hiu] = bsp->knot_vector().span_bounds(su);
        auto [lov, hiv] = bsp->knot_vector().span_bounds(sv);
        if (std::abs(hiu-lou) > 1e-14 && std::abs(hiv-lov) > 1e-14)
            ++active_elems;
    }
    const Index Q = gauss2d->num_points();
    Vector<double> load_vals = Vector<double>::Zero(active_elems * Q * ndof);
    for (Index i = 0; i < active_elems * Q; ++i)
        load_vals(i * ndof + 0) = q0;

    // ------------------------------------------------------------------
    // Reference solution: DirectConstraint clamped BCs
    // (zero w on all nodes of all edges; zero θ_x, θ_y on all edge nodes)
    // ------------------------------------------------------------------
    Vector<double> u_ref;
    {
        auto surface = make_surface();
        LinearElasticProblem<double, 2> problem(surface, element, gauss2d);
        problem.add_condition(std::make_shared<LoadCondition<double,2>>(
            *surface, *element, *gauss2d, load_vals));

        Matrix<double> K;
        Vector<double> F;
        problem.assemble(K, F);

        std::vector<Index> fixed_dofs;
        for (int dim = 0; dim < 2; ++dim)
            for (bool start : {true, false})
                for (auto node : surface->dof_mapper().get_layer_dofs(dim, start, 0))
                    for (Index v = 0; v < ndof; ++v)
                        fixed_dofs.push_back(node * ndof + v);

        std::sort(fixed_dofs.begin(), fixed_dofs.end());
        fixed_dofs.erase(std::unique(fixed_dofs.begin(), fixed_dofs.end()), fixed_dofs.end());

        DirectConstraint<double>(
            IndexVector(Eigen::Map<const IndexVector>(fixed_dofs.data(), fixed_dofs.size())),
            Vector<double>::Zero(fixed_dofs.size())).apply(K, F);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(K.sparseView());
        REQUIRE(solver.info() == Eigen::Success);
        u_ref = solver.solve(F);
    }

    // ------------------------------------------------------------------
    // Penalty solution: large α_w + α_φn + α_φs on all edges
    // ------------------------------------------------------------------
    Vector<double> u_pen;
    {
        auto surface = make_surface();
        LinearElasticProblem<double, 2> problem(surface, element, gauss2d);
        problem.add_condition(std::make_shared<LoadCondition<double,2>>(
            *surface, *element, *gauss2d, load_vals));

        const double alpha = 1e4 * D / (a * a * a);
        std::vector<Ptr<PatchBoundary<double, 2>>> penalty_boundaries;
        add_penalty_all_edges<double>(problem, penalty_boundaries, surface, *element, gauss1d,
                                      alpha, 0.0,   // w
                                      alpha, 0.0,   // rot_n
                                      alpha, 0.0);  // rot_s

        Matrix<double> K;
        Vector<double> F;
        problem.assemble(K, F);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(K.sparseView());
        REQUIRE(solver.info() == Eigen::Success);
        u_pen = solver.solve(F);
    }

    double w_ref = eval_w_centre(surface_ref, u_ref, ndof);
    double w_pen = eval_w_centre(surface_ref, u_pen, ndof);

    INFO("RM-3p clamped: w_ref=" << w_ref << "  w_pen=" << w_pen);

    // Both solutions have the same sign and order of magnitude
    REQUIRE(w_ref < 0.0);
    REQUIRE(w_pen < 0.0);

    // Penalty solution within 5 % of the DirectConstraint reference
    double rel_err = std::abs(w_pen - w_ref) / std::abs(w_ref);
    INFO("rel_err = " << rel_err);
    REQUIRE(rel_err < 5e-2);

    // Penalty deflection is strictly smaller in magnitude (less constrained
    // enforcement always allows a tiny bit more deformation)
    REQUIRE(std::abs(w_pen) >= std::abs(w_ref) * 0.95);
}

// ===========================================================================
// TEST 6 — RM-3p plate: prescribed non-zero BC
// A plate with uniform prescribed value_w != 0 on all edges, zero load.
// The expected solution is a rigid-body lift: w ≈ value_w everywhere.
// ===========================================================================
TEST_CASE("PenaltyBoundaryCondition RM-3p: prescribed non-zero displacement",
          "[conditions][penalty]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    double D  = E * h*h*h / (12.0 * (1.0 - nu*nu));
    double value_w = 2.5;

    Index p = 2, n_elem = 4;
    int   nq = p + 1;

    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    double k_shear = 5.0 / 6.0;
    auto material  = std::make_shared<PlaneStress2d<double>>(E, nu, h, k_shear);
    auto element   = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    const Index ndof = element->num_node_dofs();
    auto gauss2d   = std::make_shared<GaussLegendre<double, 2>>(nq);
    GaussLegendre<double, 1> gauss1d(nq);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);
    // No load — only prescribed displacement via penalty

    const double penalty = 1e6 * D / (a * a * a);
    std::vector<Ptr<PatchBoundary<double, 2>>> penalty_boundaries;
    add_penalty_all_edges<double>(problem, penalty_boundaries, surface, *element, gauss1d,
                                  penalty, value_w);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K.sparseView());
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    double w_centre = eval_w_centre(surface, u, ndof);
    INFO("Prescribed value_w=" << value_w << "  w_centre=" << w_centre);

    // Interior should be close to the prescribed value (plate rigid-body lift)
    double rel_err = std::abs(w_centre - value_w) / std::abs(value_w);
    REQUIRE(rel_err < 1e-2);
}

// ===========================================================================
// TEST 7 — Normal direction correctness: each boundary gets the right outward normal.
// For an axis-aligned rectangle, normal contribution to K_φn must vanish
// in the direction parallel to the boundary (where n is perpendicular).
// ===========================================================================
TEST_CASE("PenaltyBoundaryCondition normal direction: axis-aligned rectangle",
          "[conditions][penalty]")
{
    // For a boundary at x = const (param_dim=0), outward normal n = (±1, 0).
    // The φ_s contribution involves s = (0, ±1), so K_{θy, θy} from φ_s
    // should equal K_{θx, θx} from φ_n (they are both n_x^2 or s_x^2 = 0).
    //
    // Concretely: on the right edge (n=(1,0), s=(0,1)):
    //   K_φn[θx, θx] = α * 1^2 * M   K_φn[θy, θy] = α * 0^2 * M = 0
    //   K_φs[θx, θx] = α * 0^2 * M = 0   K_φs[θy, θy] = α * 1^2 * M
    //
    // So K[θx, θx] > 0 (from φ_n), K[θy, θy] > 0 (from φ_s), K[θx, θy] = 0.

    Index p = 2, n_elem = 3;
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n_elem));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, 1.0, 1.0));

    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.01);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    GaussLegendre<double, 1> g1(p + 1);

    const double penalty = 1e4;
    const Index  ndof  = 3;
    const Index  N     = surface->num_control_pts() * ndof;

    // Right edge: param_dim=0, at_start=false  →  n=(1,0), s=(0,1)
    auto bdy_r = create_patch_boundary<double, 2>(surface, 0, false);
    PenaltyBoundaryCondition<double, 2> pen_r(*bdy_r, *element, g1);
    pen_r.add(std::make_shared<NormalRotation<double>>(), penalty, 0.0);
    pen_r.add(std::make_shared<TangentialRotation<double>>(), penalty, 0.0);
    DofLayout layout_r;
    auto primal_r = layout_r.allocate(pyck::DofType::Primal, surface->num_control_pts() * ndof, ndof);
    Matrix<double> K_r = Matrix<double>::Zero(N, N);
    Vector<double> F_r = Vector<double>::Zero(N);
    pen_r.apply(K_r, F_r, layout_r, std::vector<DofLayout::BlockId>{primal_r});

    // For each boundary node: K[θx, θy] = 0 (cross-term)
    // and K[θx, θx] ≈ K[θy, θy] (same mass integral, same penalty)
    auto edge_nodes = surface->dof_mapper().get_layer_dofs(0, false, 0);
    for (auto node : edge_nodes) {
        double K_xx = K_r(node*ndof+1, node*ndof+1);
        double K_yy = K_r(node*ndof+2, node*ndof+2);
        double K_xy = K_r(node*ndof+1, node*ndof+2);

        REQUIRE(K_xx > 0.0);   // φ_n  penalises θ_x on right edge
        REQUIRE(K_yy > 0.0);   // φ_s  penalises θ_y on right edge
        REQUIRE(std::abs(K_xy) < 1e-12);  // no cross-term when n=(1,0)
    }

    // Bottom edge: param_dim=1, at_start=true  →  n=(0,-1), s=(1,0)
    auto bdy_b = create_patch_boundary<double, 2>(surface, 1, true);
    PenaltyBoundaryCondition<double, 2> pen_b(*bdy_b, *element, g1);
    pen_b.add(std::make_shared<NormalRotation<double>>(), penalty, 0.0);
    pen_b.add(std::make_shared<TangentialRotation<double>>(), penalty, 0.0);
    DofLayout layout_b;
    auto primal_b = layout_b.allocate(pyck::DofType::Primal, surface->num_control_pts() * ndof, ndof);
    Matrix<double> K_b = Matrix<double>::Zero(N, N);
    Vector<double> F_b = Vector<double>::Zero(N);
    pen_b.apply(K_b, F_b, layout_b, std::vector<DofLayout::BlockId>{primal_b});

    auto bot_nodes = surface->dof_mapper().get_layer_dofs(1, true, 0);
    for (auto node : bot_nodes) {
        double K_xx = K_b(node*ndof+1, node*ndof+1);
        double K_yy = K_b(node*ndof+2, node*ndof+2);
        double K_xy = K_b(node*ndof+1, node*ndof+2);

        REQUIRE(K_yy > 0.0);   // φ_n  penalises θ_y on bottom edge
        REQUIRE(K_xx > 0.0);   // φ_s  penalises θ_x on bottom edge
        REQUIRE(std::abs(K_xy) < 1e-12);  // no cross-term when n=(0,±1)
    }
}
