#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>
#include <numeric>

#include "patch.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knots.hpp"
#include "plate_kirchhoff_love_1p.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "direct_constraint.hpp"
#include "plane_stress_2d.hpp"
#include <Eigen/Dense>

using namespace pyck;

// ---------------------------------------------------------------------------
// Navier series solution for a simply-supported rectangular plate under
// bi-sinusoidal load
// ---------------------------------------------------------------------------

TEST_CASE("Kirchhoff-Love 1P Plate: Navier Bi-Sinusoidal Load",
          "[elements][kirchhoff_love]")
{
    // Physical parameters
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double L  = 2.0; // length
    double W  = 1.0; // width
    double f0 = -10.0; // peak intensity
    
    // Discretization parameters
    Index p = 3;
    Index n = 10;
    int   nq = p + 1; 

    auto knots_u = std::make_shared<BSpline<double>>(
        p, clamped_uniform_knots<double>(p, n));
    auto knots_v = std::make_shared<BSpline<double>>(
        p, clamped_uniform_knots<double>(p, n));

    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(knots_u, knots_v, L, W));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element = std::make_shared<PlateKirchhoffLove1p<double>>(material);
    auto gauss   = std::make_shared<GaussLegendre<double, 2>>(nq);

    LinearElasticProblem<double, 2> problem(surface, element, gauss);

    // Evaluate physical points to apply the variable load
    auto phys_pts = surface->eval_physical_points(*gauss);
    Index num_eval_pts = phys_pts.rows();
    
    Vector<double> load_vals(num_eval_pts);
    for(Index i = 0; i < num_eval_pts; ++i) {
        double x = phys_pts(i, 0);
        double y = phys_pts(i, 1);
        load_vals(i) = f0 * std::sin(M_PI * x / L) * std::sin(M_PI * y / W);
    }

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
            auto edge = surface->dof_mapper().get_layer_dofs(dim, start, 0);
            ss_dofs.insert(ss_dofs.end(), edge.begin(), edge.end());
        }
    }
    // Remove duplicates (corners appear twice)
    std::sort(ss_dofs.begin(), ss_dofs.end());
    ss_dofs.erase(std::unique(ss_dofs.begin(), ss_dofs.end()), ss_dofs.end());

    std::vector<double> ss_vals(ss_dofs.size(), 0.0);
    DirectConstraint<double> dirichlet(ss_dofs, ss_vals);
    dirichlet.apply(K, F);

    // Solve using a dense matrix solver as requested
    Eigen::PartialPivLU<Matrix<double>> solver(K);
    REQUIRE(solver.info() == Eigen::Success);
    Vector<double> u = solver.solve(F);

    // Benchmark analytical solutions
    double K_b = material->bending_stiffness();
    double term = std::pow(M_PI / L, 2) + std::pow(M_PI / W, 2);
    auto w_exact_fn = [&](double x, double y) {
        return (f0 / (K_b * std::pow(term, 2))) * std::sin(M_PI * x / L) * std::sin(M_PI * y / W);
    };

    // Evaluate and verify at several points
    std::vector<double> test_coords = {0.25, 0.5, 0.75};
    for (double pu : test_coords) {
        for (double pv : test_coords) {
            ColMatrix<double, 2> pt(1, 2);
            pt << pu, pv;
            
            Index span_u = knots_u->knot_vector().find_span(p, pu);
            Index span_v = knots_v->knot_vector().find_span(p, pv);
            auto intervals = surface->tensor_product().num_intervals();
            Index flat = span_u * intervals[1] + span_v;

            auto [sf, jac] = surface->eval_shape_functions(pt, flat, 0);
            auto active = surface->dof_mapper().get_element_dofs(flat);
            Vector<double> u_active(active.size());
            for (std::size_t i = 0; i < active.size(); ++i)
                u_active(i) = u(active[i]);

            double w_num = (sf[0] * u_active)(0, 0);
            
            // Map parametric to physical for exact solution
            double x = pu * L;
            double y = pv * W;
            double w_exact = w_exact_fn(x, y);

            double rel_err = std::abs(w_num - w_exact) / std::abs(w_exact);
            
            UNSCOPED_INFO("Point (" << x << ", " << y << ")");
            UNSCOPED_INFO("w_exact = " << w_exact);
            UNSCOPED_INFO("w_num   = " << w_num);
            REQUIRE(rel_err < 1e-3);
        }
    }

    // Calculate and verify Total Strain Energy
    double U_num = 0.5 * u.dot(K * u);
    double U_exact = (1.0 / 8.0) * (f0 * f0 * L * W) / (K_b * std::pow(term, 2));

    double rel_err_U = std::abs(U_num - U_exact) / std::abs(U_exact);
    INFO("U_exact    = " << U_exact);
    INFO("U_num      = " << U_num);
    REQUIRE(rel_err_U < 1e-3);
}
