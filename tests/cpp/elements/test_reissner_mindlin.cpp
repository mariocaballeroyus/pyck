#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>
#include <numeric>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "plate_reissner_mindlin_3p.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"
#include "linear_elastic_problem.hpp"
#include "direct_constraint.hpp"
#include "plane_stress_2d.hpp"
#include <Eigen/Dense>

using namespace pyck;

// ---------------------------------------------------------------------------
// Navier series for a simply-supported square plate under uniform load q0.
// Valid for thin plates (RM → KL as h/a → 0).
// ---------------------------------------------------------------------------
static double navier_ss(double x, double y, double a, double q0, double D,
                        int N_terms = 100)
{
    const double pi = M_PI;
    double w = 0.0;
    for (int m = 1; m <= N_terms; m += 2)
        for (int n = 1; n <= N_terms; n += 2) {
            double denom = m * n * std::pow(m * m + n * n, 2);
            w += std::sin(m * pi * x / a) * std::sin(n * pi * y / a) / denom;
        }
    return 16.0 * q0 / (std::pow(pi, 6) * D) * w;
}

// ---------------------------------------------------------------------------
// Solve a SS square RM plate under uniform load q0 and return u (full DOF vec).
// ---------------------------------------------------------------------------
static Eigen::VectorXd solve_ss_rm_plate(
    Index degree, Index num_basis_1d, double a, double q0,
    double E, double nu, double h, int n_quad)
{
    auto bsp = std::make_shared<BSpline<double>>(
        degree, KnotVector<double>::clamped_uniform(degree, num_basis_1d));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    // 2. Setup Material and Element
    double k  = 5.0 / 6.0;
    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h, k);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss    = std::make_shared<GaussLegendre<double, 2>>(n_quad);

    LinearElasticProblem<double, 2> problem(surface, element, gauss);

    problem.add_domain_load(*surface, q0);

    SparseMatrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    // SS BCs: w = 0 (only DOF 0 of 3 per node) at all four edges
    const Index ndof = element->num_node_dofs();
    std::vector<Index> ss_dofs;
    for (int dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto edge = surface->dof_mapper().get_layer_dofs(dim, start, 0);
            for (auto node : edge)
                ss_dofs.push_back(node * ndof + 0);  // w-DOF only
        }
    }
    std::sort(ss_dofs.begin(), ss_dofs.end());
    ss_dofs.erase(std::unique(ss_dofs.begin(), ss_dofs.end()), ss_dofs.end());

    DirectConstraint<double>(
        IndexVector(Eigen::Map<const IndexVector>(ss_dofs.data(), ss_dofs.size())),
        Vector<double>::Zero(ss_dofs.size())).apply(K, F);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);
    REQUIRE(solver.info() == Eigen::Success);
    return solver.solve(F);
}

TEST_CASE("Reissner-Mindlin Plate element properties", "[RMPlate]")
{
    auto knots = std::make_shared<BSpline<double>>(
        1, KnotVector<double>::clamped_uniform(1, 2)); // Linear spline
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(knots, knots, 1.0, 1.0));

    double E = 200e9;
    double nu = 0.3;
    double h = 0.01;
    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    PlateReissnerMindlin3p<double> element(material);

    SECTION("Basic properties") {
        REQUIRE(element.num_node_dofs() == 3);
        REQUIRE(element.basis_order() == 2);
    }

    SECTION("Constitutive matrices") {
        // Just verify they are created and have correct size
        // Bending D is ~ E*h^3 / (12*(1-nu^2))
        // Shear D is k*G*h
    }
}

TEST_CASE("Reissner-Mindlin Plate Stiffness Matrix Size", "[RMPlate]")
{
    // p=2, 3x3 basis -> 9 nodes
    Index p = 2;
    Index n = 3;
    auto knots = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    Index num_pts = n * n;
    ColMatrix<double, 3> P(num_pts, 3);
    P.setZero();
    for (Index iv = 0; iv < n; ++iv) {
        for (Index iu = 0; iu < n; ++iu) {
            P.row(iu + iv * n) << (double)iu / (n - 1), (double)iv / (n - 1), 0.0;
        }
    }
    auto surface = std::make_shared<Patch<double, 2>>(knots, knots, P);

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.1);
    PlateReissnerMindlin3p<double> element(material);

    // Single-point Gauss on the patch's only non-zero span (live element 0).
    GaussLegendre<double, 2> gauss(1);
    ElementValues<double, 2> pv(*surface,
                              element.basis_order(), element.flags(), gauss);
    pv.reinit(0);

    Matrix<double> stiffness;
    element.compute_local_stiffness(pv, stiffness);

    // 9 nodes * 3 DOFs/node = 27
    REQUIRE(stiffness.rows() == 27);
    REQUIRE(stiffness.cols() == 27);
    
    // Symmetry check
    REQUIRE( stiffness.allFinite() );
    REQUIRE( (stiffness - stiffness.transpose()).norm() < 1e-12 );
}

// ===========================================================================
// TEST 3 — SS plate: centre deflection agrees with Navier series (thin plate)
// ===========================================================================
TEST_CASE("RM Plate: SS uniform load — thin plate convergence", "[RMPlate]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;   // thin: a/h = 100 → RM ≈ KL
    double a  = 1.0;
    double q0 = -1.0;
    double D  = E * h * h * h / (12.0 * (1.0 - nu * nu));

    double w_exact = navier_ss(a / 2.0, a / 2.0, a, q0, D, 200);

    Index p = 3;
    Index n = 8;
    auto u = solve_ss_rm_plate(p, n, a, q0, E, nu, h, p + 1);

    // Evaluate w at centre (xi = eta = 0.5)
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surf = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    ColMatrix<double, 2> pt(1, 2);
    pt << 0.5, 0.5;
    Index span_u = bsp->knot_vector().find_span(p, 0.5);
    Index span_v = bsp->knot_vector().find_span(p, 0.5);
    auto intervals = surf->tensor_product().num_intervals();
    Index flat = span_u + span_v * intervals[0];   // u-fastest

    const auto b = (*surf).tensor_product().eval_all(pt, 0);
    std::vector<Index> active;
    surf->dof_mapper().get_element_cps(flat, active);

    const Index ndof = 3;
    Vector<double> w_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        w_active(i) = u(active[i] * ndof + 0);  // w-DOF

    double w_num = b[0].col(0).dot(w_active);
    double rel_err = std::abs(w_num - w_exact) / std::abs(w_exact);

    INFO("w_exact = " << w_exact);
    INFO("w_num   = " << w_num);
    INFO("rel_err = " << rel_err);

    REQUIRE(rel_err < 0.02);  // < 2 % for p=3, n=8, thin plate
}

// ===========================================================================
// TEST 4 — Thick plate: RM non-dim coefficient > thin-plate KL value
// ===========================================================================
TEST_CASE("RM Plate: thick plate — RM > KL deflection coefficient", "[RMPlate]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double a  = 1.0;
    double q0 = -1.0;

    Index p = 3;
    Index n = 8;

    double h_thin  = 0.01;   // a/h = 100
    double h_thick = 0.1;    // a/h = 10  — shear deformation adds up

    auto u_thin  = solve_ss_rm_plate(p, n, a, q0, E, nu, h_thin,  p + 1);
    auto u_thick = solve_ss_rm_plate(p, n, a, q0, E, nu, h_thick, p + 1);

    // Evaluate w at centre from each solution (same mesh → same shape fns)
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surf = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    ColMatrix<double, 2> pt(1, 2);
    pt << 0.5, 0.5;
    Index span_u = bsp->knot_vector().find_span(p, 0.5);
    Index span_v = bsp->knot_vector().find_span(p, 0.5);
    auto intervals = surf->tensor_product().num_intervals();
    Index flat = span_u + span_v * intervals[0];   // u-fastest

    const auto b = (*surf).tensor_product().eval_all(pt, 0);
    std::vector<Index> active;
    surf->dof_mapper().get_element_cps(flat, active);

    const Index ndof = 3;
    auto get_w = [&](const Eigen::VectorXd& u) {
        Vector<double> wa(active.size());
        for (std::size_t i = 0; i < active.size(); ++i)
            wa(i) = u(active[i] * ndof + 0);
        return b[0].col(0).dot(wa);
    };

    double w_thin  = get_w(u_thin);
    double w_thick = get_w(u_thick);

    // Non-dimensionalise: alpha = w * D / (q0 * a^4)
    auto D = [&](double h) {
        return E * h * h * h / (12.0 * (1.0 - nu * nu));
    };
    double alpha_thin  = w_thin  * D(h_thin)  / (q0 * a * a * a * a);
    double alpha_thick = w_thick * D(h_thick) / (q0 * a * a * a * a);

    INFO("alpha_thin  (≈0.00406) = " << alpha_thin);
    INFO("alpha_thick (> alpha_thin) = " << alpha_thick);

    // For the thick plate the shear correction adds to the deflection
    REQUIRE(alpha_thick > alpha_thin);
}

