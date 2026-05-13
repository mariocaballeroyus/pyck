#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include "patch.hpp"
#include "basis_derivs.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knots.hpp"
#include "beam_euler_bernoulli_1p.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "direct_constraint.hpp"
#include "uniaxial_stress_1d.hpp"
#include "plane_stress_2d.hpp"
#include <Eigen/Dense>

using namespace pyck;

TEST_CASE("Euler-Bernoulli Beam: Simply Supported Uniform Load", "[assembly][euler_bernoulli]") {
    
    // Physical Parameters
    double E = 200e9; // Young's modulus (Pa) -> Steel
    double I = 1e-4;  // Second moment of area (m^4)
    double A = 1e-2;  // Cross-sectional area (m^2)
    double L = 10.0;  // Length of the beam (m)
    double q = -5000;  // Uniform distributed load (N/m) downward

    // 1. Geometry: Single patch line from x=0 to x=L
    // A simply supported Euler-Bernoulli beam under uniform load has a quartic O(x^4) deflection profile.
    // For exact representation, we need at least p=4 (Quartic BSplines).
    std::vector<double> knots_vec = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    auto basis = std::make_shared<BSpline<double>>(4, KnotVector<double>(knots_vec));
    
    Index num_pts = basis->num_basis(); // 5 control points
    ColMatrix<double, 3> control_pts(num_pts, 3);
    control_pts.setZero();
    control_pts(0, 0) = 0.0;
    control_pts(1, 0) = L / 4.0;
    control_pts(2, 0) = 2.0 * L / 4.0;
    control_pts(3, 0) = 3.0 * L / 4.0;
    control_pts(4, 0) = L;

    auto curve = std::make_shared<Patch<double, 1>>(basis, control_pts);

    auto material = std::make_shared<UniaxialStress1d<double>>(E, 0.3, A, I);

    // 2. Element & Quadrature
    auto beam_elem = std::make_shared<BeamEulerBernoulli1p<double>>(material);
    // Quartic basis requires 5 points for exact bending integral (order 2p-2 = 6, 4 pts gives degree 7)
    auto gauss_rule = std::make_shared<GaussLegendre<double, 1>>(5);

    LinearElasticProblem<double, 1> problem(curve, beam_elem, gauss_rule);

    // 4. Conditions: Uniform Dist Load (1 element, 5 quad points = 5 total)
    Vector<double> load_values = Vector<double>::Constant(5, q);
    auto load_cond = std::make_shared<LoadCondition<double, 1>>(*curve, *beam_elem, *gauss_rule, load_values);
    problem.add_condition(load_cond);

    // 5. Assemble Global System
    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    // 6. Apply Essential Boundary Conditions (Simply Supported)
    // - Left end (x=0) constrained vertically (v = 0). For 1p beam, DOFs represent transversal disp.
    // - Right end (x=L) constrained vertically (v = 0).
    // In our single patch scalar mapping, DOF 0 is at x=0, and DOF 4 is at x=L.
    std::vector<Index> constrained_dofs = {0, 4}; 
    std::vector<double> constrained_vals = {0.0, 0.0};
    
    DirectConstraint<double> dirichlet(constrained_dofs, constrained_vals);
    dirichlet.apply(K, F);

    // 7. Solve Ku = F
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    // We dense-to-sparse cast for the solver
    Eigen::SparseMatrix<double> K_sparse = K.sparseView();
    solver.compute(K_sparse);
    
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    // 8. Validation
    // The exact analytical mid-span deflection of a simply supported beam is:
    // v_max = (5 * q * L^4) / (384 * E * I)
    double v_exact = (5.0 * q * std::pow(L, 4)) / (384.0 * E * I);
    
    // Evaluate the numerical displacement at mid-span (ξ = 0.5)
    ColMatrix<double, 1> param(1, 1);
    param(0, 0) = 0.5;
    Index span = basis->find_span(0.5);
    const auto b = eval_basis(*curve, param, span, 0);

    auto active = curve->dof_mapper().get_element_dofs(span);
    Vector<double> u_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        u_active(i) = u(active[i]);
    double v_num = (b.N * u_active)(0, 0);

    // Strain energy
    // U = 1/2 * u^T * K * u
    double U_num = 0.5 * u.dot(K * u);
    double U_exact = (q * q * std::pow(L, 5)) / (240.0 * E * I);

    // Note: Due to 1p beam approximation and scalar DOF mapping, there might be slight
    // deviations compared to a full Timoshenko implementation, but for a cubic
    // spline it should be capable of recovering the exact polynomial!
    REQUIRE(v_num == Approx(v_exact).margin(1e-16));
    REQUIRE(U_num == Approx(U_exact).margin(1e-16));
}

TEST_CASE("Euler-Bernoulli Beam: Simply Supported Uniform Load (Cubic Approximation)", "[assembly][euler_bernoulli]") {
    
    // Physical Parameters
    double E = 200e9; // Young's modulus (Pa) -> Steel
    double I = 1e-4;  // Second moment of area (m^4)
    double A = 1e-2;  // Cross-sectional area (m^2)
    double L = 10.0;  // Length of the beam (m)
    double q = -5000;  // Uniform distributed load (N/m) downward

    // Geometry: Single patch line from x=0 to x=L
    // A simply supported Euler-Bernoulli beam under uniform load has a quartic O(x^4) deflection profile.
    // By providing a cubic basis (p=3), we force the system to approximate the solution.
    // However, by inserting an intermediate knot at ξ=0.5, we split the domain into two 
    // cubic elements, significantly improving the approximation accuracy.
    std::vector<double> knots_vec = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0};
    auto basis = std::make_shared<BSpline<double>>(3, KnotVector<double>(knots_vec));
    
    Index num_pts = basis->num_basis(); // 5 control points
    ColMatrix<double, 3> control_pts(num_pts, 3);
    control_pts.setZero();
    control_pts(0, 0) = 0.0;
    control_pts(1, 0) = L / 4.0;
    control_pts(2, 0) = L / 2.0;
    control_pts(3, 0) = 3.0 * L / 4.0;
    control_pts(4, 0) = L;

    auto curve = std::make_shared<Patch<double, 1>>(basis, control_pts);

    // Element & Quadrature
    auto material = std::make_shared<UniaxialStress1d<double>>(E, 0.3, A, I);
    auto element = std::make_shared<BeamEulerBernoulli1p<double>>(material);
    // Cubic basis requires 4 points for exact bending integral (order 2p-2 = 4)
    auto gauss_rule = std::make_shared<GaussLegendre<double, 1>>(4);

    LinearElasticProblem<double, 1> problem(curve, element, gauss_rule);

    // Conditions: Uniform Dist Load (2 elements, 4 quad points each = 8 total)
    Vector<double> load_values = Vector<double>::Constant(8, q);
    auto load_cond = std::make_shared<LoadCondition<double, 1>>(*curve, *element, *gauss_rule, load_values);
    problem.add_condition(load_cond);

    // Assemble Global System
    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    // Apply Essential Boundary Conditions (Simply Supported)
    std::vector<Index> constrained_dofs = {0, 4}; 
    std::vector<double> constrained_vals = {0.0, 0.0};
    
    DirectConstraint<double> dirichlet(constrained_dofs, constrained_vals);
    dirichlet.apply(K, F);

    // 7. Solve Ku = F
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> K_sparse = K.sparseView();
    solver.compute(K_sparse);
    
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    // 8. Validation
    double v_exact = (5.0 * q * std::pow(L, 4)) / (384.0 * E * I);
    
    ColMatrix<double, 1> param(1, 1);
    param(0, 0) = 0.5;
    Index span2 = basis->find_span(0.5);
    const auto b = eval_basis(*curve, param, span2, 0);

    auto active = curve->dof_mapper().get_element_dofs(span2);
    Vector<double> u_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        u_active(i) = u(active[i]);
    double v_num = (b.N * u_active)(0, 0);

    // Strain energy
    double U_num = 0.5 * u.dot(K * u);
    double U_exact = (q * q * std::pow(L, 5)) / (240.0 * E * I);

    // Verify the approximation has a measurable but bounded deviation
    REQUIRE(v_num != Approx(v_exact).margin(1e-12));
    REQUIRE(U_num != Approx(U_exact).margin(1e-12));

    // With 2 cubic elements, the error should drop from ~20% to ~1%
    REQUIRE(v_num == Approx(v_exact).epsilon(0.015));
    REQUIRE(U_num == Approx(U_exact).epsilon(0.015));
}
