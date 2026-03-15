#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>
#include <numeric>

#include "surface.hpp"
#include "bspline.hpp"
#include "knots.hpp"
#include "kirchhoff_love_plate.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "direct_constraint.hpp"
#include "slender_beam_1d.hpp"
#include "plane_stress_2d.hpp"
#include <Eigen/Dense>

using namespace pyck;

// ---------------------------------------------------------------------------
// Navier series solution for a simply-supported rectangular plate under
// uniform transverse load q0.
//
//   w(x,y) = (16 q0) / (pi^6 D) *
//            sum_{m=1,3,...} sum_{n=1,3,...}
//              sin(m pi x / a) sin(n pi y / b)
//              / [ m n (m^2/a^2 + n^2/b^2)^2 ]
// ---------------------------------------------------------------------------
static double navier_ss(double x, double y, double a, double b,
                        double q0, double D, int N_terms = 100)
{
    const double pi = M_PI;
    double w = 0.0;
    for (int m = 1; m <= N_terms; m += 2) {
        for (int n = 1; n <= N_terms; n += 2) {
            double denom = m * n
                * std::pow(m * m / (a * a) + n * n / (b * b), 2);
            w += std::sin(m * pi * x / a) * std::sin(n * pi * y / b) / denom;
        }
    }
    return 16.0 * q0 / (std::pow(pi, 6) * D) * w;
}

// ---------------------------------------------------------------------------
// Helper: build a simply-supported square plate problem and solve it.
// Returns the solution vector u (one DOF per control point).
// ---------------------------------------------------------------------------
static Eigen::VectorXd solve_ss_plate(
    Index degree, Index num_basis_1d, double a, double q0,
    double E, double nu, double h, int n_quad)
{
    auto knots = std::make_shared<BSpline<double>>(
        degree, clamped_uniform_knots<double>(degree, num_basis_1d));
    auto surface = std::make_shared<SurfacePatch<double>>(
        rectangle<double>(knots, knots, a, a));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element = std::make_shared<KirchhoffLovePlate<double>>(material);
    auto gauss   = std::make_shared<GaussLegendre<double, 2>>(n_quad);

    LinearElasticProblem<double, 2> problem(surface, element);
    problem.set_quadrature(gauss);

    // Uniform load: all quadrature points get the same value q0
    auto intervals = surface->tensor_product().num_intervals();
    Index total_elements = intervals[0] * intervals[1];
    // Count active (non-zero-volume) elements to size the load vector
    Index active_elements = 0;
    for (Index e = 0; e < total_elements; ++e) {
        std::array<Index, 2> spans;
        spans[1] = e % intervals[1];
        spans[0] = e / intervals[1];
        bool zero = false;
        for (int i = 0; i < 2; ++i) {
            auto [lo, hi] = knots->knot_vector().span_bounds(spans[i]);
            if (std::abs(hi - lo) < 1e-14) { zero = true; break; }
        }
        if (!zero) ++active_elements;
    }
    Index Q = gauss->num_points();
    Vector<double> load_vals = Vector<double>::Constant(active_elements * Q, q0);

    auto load_cond = std::make_shared<LoadCondition<double, 2>>(
        *surface, *element, *gauss, load_vals);
    problem.add_condition(load_cond);

    // Assemble
    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    // Simply-supported BCs: w = 0 on all four edges
    std::vector<Index> ss_dofs;
    for (int dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto edge = surface->dof_mapper().get_displacement_boundary_dofs(dim, start);
            ss_dofs.insert(ss_dofs.end(), edge.begin(), edge.end());
        }
    }
    // Remove duplicates (corners appear twice)
    std::sort(ss_dofs.begin(), ss_dofs.end());
    ss_dofs.erase(std::unique(ss_dofs.begin(), ss_dofs.end()), ss_dofs.end());

    std::vector<double> ss_vals(ss_dofs.size(), 0.0);
    DirectConstraint<double> dirichlet(ss_dofs, ss_vals);
    dirichlet.apply(K, F);

    // Solve
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> Ks = K.sparseView();
    solver.compute(Ks);
    REQUIRE(solver.info() == Eigen::Success);
    return solver.solve(F);
}


// ===========================================================================
// TEST 1 — Simply Supported Plate, Uniform Load (convergence)
// ===========================================================================
TEST_CASE("KL Plate: SS Uniform Load — Centre Deflection",
          "[assembly][kirchhoff_love]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    double q0 = -1.0;
    double D  = E * h * h * h / (12.0 * (1.0 - nu * nu));

    double w_exact = navier_ss(a / 2, a / 2, a, a, q0, D, 200);

    // p = 3, 8 basis functions per direction, 4 quad points
    Index p = 3;
    Index n = 8;
    int   nq = p + 1;

    auto u = solve_ss_plate(p, n, a, q0, E, nu, h, nq);

    // Evaluate at centre (u = v = 0.5)
    auto knots_obj = std::make_shared<BSpline<double>>(
        p, clamped_uniform_knots<double>(p, n));
    auto surf = std::make_shared<SurfacePatch<double>>(
        rectangle<double>(knots_obj, knots_obj, a, a));

    ColMatrix<double, 2> pt(1, 2);
    pt << 0.5, 0.5;
    // Find the element containing the centre
    Index span_u = knots_obj->knot_vector().find_span(p, 0.5);
    Index span_v = knots_obj->knot_vector().find_span(p, 0.5);
    auto intervals = knots_obj->knot_vector().num_spans();
    Index flat = span_u * intervals + span_v;

    auto [sf, jac] = surf->eval_shape_functions(pt, flat, 0);
    auto active = surf->dof_mapper().get_element_dofs(flat);
    Vector<double> u_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        u_active(i) = u(active[i]);

    double w_num = (sf[0] * u_active)(0, 0);
    double rel_err = std::abs(w_num - w_exact) / std::abs(w_exact);

    INFO("w_exact  = " << w_exact);
    INFO("w_num    = " << w_num);
    INFO("rel_err  = " << rel_err);

    // With p=3, 8 DOFs per direction we expect < 1% error
    REQUIRE(rel_err < 0.01);
}


// ===========================================================================
// TEST 2 — Simply Supported Plate, Strain Energy Check
// ===========================================================================
TEST_CASE("KL Plate: SS Uniform Load — Strain Energy Convergence",
          "[assembly][kirchhoff_love]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    double q0 = -1.0;
    double D  = E * h * h * h / (12.0 * (1.0 - nu * nu));

    // Exact strain energy by Navier series:
    //   U = (1/2) q0^2 / (pi^8 D) * 256 *
    //       sum_{m,n odd} 1 / [m^2 n^2 (m^2+n^2)^2]   (for square plate a=b)
    // More easily: U = (1/2) u^T K u  with a fine-mesh reference.
    // Use a fine solution as reference.
    auto u_fine = solve_ss_plate(4, 16, a, q0, E, nu, h, 5);

    // Recompute K for the fine mesh to get reference energy
    Index p_f = 4, n_f = 16;
    auto knots_f = std::make_shared<BSpline<double>>(
        p_f, clamped_uniform_knots<double>(p_f, n_f));
    auto surf_f = std::make_shared<SurfacePatch<double>>(
        rectangle<double>(knots_f, knots_f, a, a));
    auto mat_f = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto elem_f = std::make_shared<KirchhoffLovePlate<double>>(mat_f);
    auto gauss_f = std::make_shared<GaussLegendre<double, 2>>(5);

    LinearElasticProblem<double, 2> prob_f(surf_f, elem_f);
    prob_f.set_quadrature(gauss_f);

    auto intervals_f = surf_f->tensor_product().num_intervals();
    Index total_f = intervals_f[0] * intervals_f[1];
    Index active_f = 0;
    for (Index e = 0; e < total_f; ++e) {
        std::array<Index, 2> spans;
        spans[1] = e % intervals_f[1];
        spans[0] = e / intervals_f[1];
        bool zero = false;
        for (int i = 0; i < 2; ++i) {
            auto [lo, hi] = knots_f->knot_vector().span_bounds(spans[i]);
            if (std::abs(hi - lo) < 1e-14) { zero = true; break; }
        }
        if (!zero) ++active_f;
    }
    Vector<double> lv_f = Vector<double>::Constant(active_f * gauss_f->num_points(), q0);
    auto lc_f = std::make_shared<LoadCondition<double, 2>>(
        *surf_f, *elem_f, *gauss_f, lv_f);
    prob_f.add_condition(lc_f);

    Matrix<double> K_f;
    Vector<double> F_f;
    prob_f.assemble(K_f, F_f);

    // Apply BCs
    std::vector<Index> ss_dofs_f;
    for (int dim = 0; dim < 2; ++dim)
        for (bool start : {true, false}) {
            auto edge = surf_f->dof_mapper().get_displacement_boundary_dofs(dim, start);
            ss_dofs_f.insert(ss_dofs_f.end(), edge.begin(), edge.end());
        }
    std::sort(ss_dofs_f.begin(), ss_dofs_f.end());
    ss_dofs_f.erase(std::unique(ss_dofs_f.begin(), ss_dofs_f.end()), ss_dofs_f.end());
    DirectConstraint<double>(ss_dofs_f, std::vector<double>(ss_dofs_f.size(), 0.0)).apply(K_f, F_f);

    double U_ref = 0.5 * u_fine.dot(K_f * u_fine);

    // Coarse solution: p=3, 6 DOFs per direction
    auto u_coarse = solve_ss_plate(3, 6, a, q0, E, nu, h, 4);

    auto knots_c = std::make_shared<BSpline<double>>(
        3, clamped_uniform_knots<double>(3, 6));
    auto surf_c = std::make_shared<SurfacePatch<double>>(
        rectangle<double>(knots_c, knots_c, a, a));
    auto mat_c = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto elem_c = std::make_shared<KirchhoffLovePlate<double>>(mat_c);
    auto gauss_c = std::make_shared<GaussLegendre<double, 2>>(4);
    LinearElasticProblem<double, 2> prob_c(surf_c, elem_c);
    prob_c.set_quadrature(gauss_c);

    auto intervals_c = surf_c->tensor_product().num_intervals();
    Index total_c = intervals_c[0] * intervals_c[1];
    Index active_c = 0;
    for (Index e = 0; e < total_c; ++e) {
        std::array<Index, 2> spans;
        spans[1] = e % intervals_c[1];
        spans[0] = e / intervals_c[1];
        bool zero = false;
        for (int i = 0; i < 2; ++i) {
            auto [lo, hi] = knots_c->knot_vector().span_bounds(spans[i]);
            if (std::abs(hi - lo) < 1e-14) { zero = true; break; }
        }
        if (!zero) ++active_c;
    }
    Vector<double> lv_c = Vector<double>::Constant(active_c * gauss_c->num_points(), q0);
    auto lc_c = std::make_shared<LoadCondition<double, 2>>(
        *surf_c, *elem_c, *gauss_c, lv_c);
    prob_c.add_condition(lc_c);

    Matrix<double> K_c;
    Vector<double> F_c;
    prob_c.assemble(K_c, F_c);

    std::vector<Index> ss_dofs_c;
    for (int dim = 0; dim < 2; ++dim)
        for (bool start : {true, false}) {
            auto edge = surf_c->dof_mapper().get_displacement_boundary_dofs(dim, start);
            ss_dofs_c.insert(ss_dofs_c.end(), edge.begin(), edge.end());
        }
    std::sort(ss_dofs_c.begin(), ss_dofs_c.end());
    ss_dofs_c.erase(std::unique(ss_dofs_c.begin(), ss_dofs_c.end()), ss_dofs_c.end());
    DirectConstraint<double>(ss_dofs_c, std::vector<double>(ss_dofs_c.size(), 0.0)).apply(K_c, F_c);

    double U_coarse = 0.5 * u_coarse.dot(K_c * u_coarse);

    // Coarseness leads to a stiffer response → U_coarse < U_ref
    double rel = std::abs(U_coarse - U_ref) / std::abs(U_ref);
    INFO("U_ref    = " << U_ref);
    INFO("U_coarse = " << U_coarse);
    INFO("rel_err  = " << rel);

    // Coarse p=3 n=6 should be within ~5% of the fine reference
    REQUIRE(rel < 0.05);
}


// ===========================================================================
// TEST 3 — Clamped Plate (all edges), Uniform Load
// ===========================================================================
TEST_CASE("KL Plate: Clamped All Edges — Centre Deflection",
          "[assembly][kirchhoff_love]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    double q0 = -1.0;
    double D  = E * h * h * h / (12.0 * (1.0 - nu * nu));

    // Known coefficient for a square clamped plate:
    //   w_max = alpha * q0 * a^4 / D,  alpha ≈ 0.00126
    double w_ref = 0.00126 * q0 * a * a * a * a / D;

    Index p = 3;
    Index n = 10;
    int   nq = p + 1;

    auto knots = std::make_shared<BSpline<double>>(
        p, clamped_uniform_knots<double>(p, n));
    auto surface = std::make_shared<SurfacePatch<double>>(
        rectangle<double>(knots, knots, a, a));
    auto mat = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element = std::make_shared<KirchhoffLovePlate<double>>(mat);
    auto gauss   = std::make_shared<GaussLegendre<double, 2>>(nq);

    LinearElasticProblem<double, 2> problem(surface, element);
    problem.set_quadrature(gauss);

    auto intervals = surface->tensor_product().num_intervals();
    Index total_elements = intervals[0] * intervals[1];
    Index active = 0;
    for (Index e = 0; e < total_elements; ++e) {
        std::array<Index, 2> spans;
        spans[1] = e % intervals[1];
        spans[0] = e / intervals[1];
        bool zero = false;
        for (int i = 0; i < 2; ++i) {
            auto [lo, hi] = knots->knot_vector().span_bounds(spans[i]);
            if (std::abs(hi - lo) < 1e-14) { zero = true; break; }
        }
        if (!zero) ++active;
    }
    Index Q = gauss->num_points();
    Vector<double> load_vals = Vector<double>::Constant(active * Q, q0);

    auto load_cond = std::make_shared<LoadCondition<double, 2>>(
        *surface, *element, *gauss, load_vals);
    problem.add_condition(load_cond);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    // Clamped BCs: w = 0 AND dw/dn = 0 on all edges
    // This means both displacement DOFs and rotation DOFs on all four edges
    std::vector<Index> clamp_dofs;
    for (int dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto edge = surface->dof_mapper().get_boundary_dofs(dim, start);
            clamp_dofs.insert(clamp_dofs.end(), edge.begin(), edge.end());
        }
    }
    std::sort(clamp_dofs.begin(), clamp_dofs.end());
    clamp_dofs.erase(std::unique(clamp_dofs.begin(), clamp_dofs.end()),
                     clamp_dofs.end());

    std::vector<double> clamp_vals(clamp_dofs.size(), 0.0);
    DirectConstraint<double> dirichlet(clamp_dofs, clamp_vals);
    dirichlet.apply(K, F);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> Ks = K.sparseView();
    solver.compute(Ks);
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    // Evaluate at centre
    ColMatrix<double, 2> pt(1, 2);
    pt << 0.5, 0.5;
    Index span_u = knots->knot_vector().find_span(p, 0.5);
    Index span_v = knots->knot_vector().find_span(p, 0.5);
    auto num_intervals = knots->knot_vector().num_spans();
    Index flat = span_u * num_intervals + span_v;

    auto [sf, jac] = surface->eval_shape_functions(pt, flat, 0);
    auto act = surface->dof_mapper().get_element_dofs(flat);
    Vector<double> u_act(act.size());
    for (std::size_t i = 0; i < act.size(); ++i)
        u_act(i) = u(act[i]);

    double w_num = (sf[0] * u_act)(0, 0);
    double rel_err = std::abs(w_num - w_ref) / std::abs(w_ref);

    INFO("w_ref   = " << w_ref);
    INFO("w_num   = " << w_num);
    INFO("rel_err = " << rel_err);

    // Known coefficient has some uncertainty; accept ~5%
    REQUIRE(rel_err < 0.05);
}


// ===========================================================================
// TEST 4 — Mixed BCs: Two edges simply-supported, two edges clamped
// ===========================================================================
TEST_CASE("KL Plate: Mixed BCs (SS/Clamped) — Symmetry Check",
          "[assembly][kirchhoff_love]")
{
    // Plate clamped at u=0 and u=1, simply supported at v=0 and v=1.
    // Under uniform load the deflection must be symmetric about
    // u = a/2 and v = a/2 (the geometry and load are symmetric).
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    auto mat = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element = std::make_shared<KirchhoffLovePlate<double>>(mat);
    double q0 = -1.0;

    Index p = 3;
    Index n = 8;

    auto knots = std::make_shared<BSpline<double>>(
        p, clamped_uniform_knots<double>(p, n));
    auto surface = std::make_shared<SurfacePatch<double>>(
        rectangle<double>(knots, knots, a, a));
    auto gauss   = std::make_shared<GaussLegendre<double, 2>>(p + 1);

    LinearElasticProblem<double, 2> problem(surface, element);
    problem.set_quadrature(gauss);

    auto intervals = surface->tensor_product().num_intervals();
    Index total = intervals[0] * intervals[1];
    Index active = 0;
    for (Index e = 0; e < total; ++e) {
        std::array<Index, 2> spans;
        spans[1] = e % intervals[1];
        spans[0] = e / intervals[1];
        bool zero = false;
        for (int i = 0; i < 2; ++i) {
            auto [lo, hi] = knots->knot_vector().span_bounds(spans[i]);
            if (std::abs(hi - lo) < 1e-14) { zero = true; break; }
        }
        if (!zero) ++active;
    }
    Vector<double> lv = Vector<double>::Constant(active * gauss->num_points(), q0);
    auto lc = std::make_shared<LoadCondition<double, 2>>(
        *surface, *element, *gauss, lv);
    problem.add_condition(lc);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    // Clamped at u=0 and u=1 (param_dim=0): both layers
    std::vector<Index> bc_dofs;
    for (bool start : {true, false}) {
        auto edge = surface->dof_mapper().get_boundary_dofs(0, start);
        bc_dofs.insert(bc_dofs.end(), edge.begin(), edge.end());
    }
    // Simply supported at v=0 and v=1 (param_dim=1): displacement only
    for (bool start : {true, false}) {
        auto edge = surface->dof_mapper().get_displacement_boundary_dofs(1, start);
        bc_dofs.insert(bc_dofs.end(), edge.begin(), edge.end());
    }
    std::sort(bc_dofs.begin(), bc_dofs.end());
    bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());

    DirectConstraint<double>(bc_dofs, std::vector<double>(bc_dofs.size(), 0.0)).apply(K, F);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> Ks = K.sparseView();
    solver.compute(Ks);
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    // Check symmetry: w(0.25, 0.5) should equal w(0.75, 0.5)
    auto eval_w = [&](double pu, double pv) {
        ColMatrix<double, 2> pt(1, 2);
        pt << pu, pv;
        Index su = knots->knot_vector().find_span(p, pu);
        Index sv = knots->knot_vector().find_span(p, pv);
        auto num_int = knots->knot_vector().num_spans();
        Index flat = su * num_int + sv;
        auto [sf, jac] = surface->eval_shape_functions(pt, flat, 0);
        auto act = surface->dof_mapper().get_element_dofs(flat);
        Vector<double> u_act(act.size());
        for (std::size_t i = 0; i < act.size(); ++i)
            u_act(i) = u(act[i]);
        return (sf[0] * u_act)(0, 0);
    };

    double w1 = eval_w(0.25, 0.5);
    double w2 = eval_w(0.75, 0.5);
    REQUIRE(w1 == Approx(w2).epsilon(1e-10));

    double w3 = eval_w(0.5, 0.25);
    double w4 = eval_w(0.5, 0.75);
    REQUIRE(w3 == Approx(w4).epsilon(1e-10));

    // The centre deflection should be smaller than the SS case
    // but non-zero
    double w_center = eval_w(0.5, 0.5);
    REQUIRE(w_center < 0.0);   // downward for negative q0
}
