#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>
#include <numeric>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "plate_reissner_mindlin_1p.hpp"
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
// Navier series for a simply-supported Reissner-Mindlin square plate
// under uniform load q0.  Includes the shear correction term.
//
// The bending potential w_b satisfies D*nabla^4(w_b) = q with w_b=0 and
// w_{b,nn}=0 on the boundary. Physical deflection:
//   w_mn = Q_mn / (D pi^4 alpha^2) + Q_mn / (Ks pi^2 alpha)
// where Q_mn = 16 q0 / (pi^2 m n) and alpha_mn = (m^2+n^2)/a^2.
// ---------------------------------------------------------------------------
static double navier_rm_ss(double x, double y, double a, double q0,
                           double D, double Ks, int N_terms = 200)
{
    const double pi = M_PI;
    double w = 0.0;
    for (int m = 1; m <= N_terms; m += 2) {
        for (int n = 1; n <= N_terms; n += 2) {
            double alpha_mn = (m * m + n * n) / (a * a);
            double pi2a = pi * pi * alpha_mn;
            double Q_mn = 16.0 * q0 / (pi * pi * m * n);
            double w_mn = Q_mn * (1.0 / (D * pi2a * pi2a) + 1.0 / (Ks * pi2a));
            w += w_mn * std::sin(m * pi * x / a) * std::sin(n * pi * y / a);
        }
    }
    return w;
}

// ---------------------------------------------------------------------------
// Navier series for KL (no shear).
// ---------------------------------------------------------------------------
static double navier_kl_ss(double x, double y, double a, double q0,
                           double D, int N_terms = 200)
{
    const double pi = M_PI;
    double w = 0.0;
    for (int m = 1; m <= N_terms; m += 2) {
        for (int n = 1; n <= N_terms; n += 2) {
            double denom = m * n * std::pow(m * m + n * n, 2);
            w += std::sin(m * pi * x / a) * std::sin(n * pi * y / a) / denom;
        }
    }
    return 16.0 * q0 / (std::pow(pi, 6) * D) * w;
}

// ---------------------------------------------------------------------------
// Navier-based exact strain energy for SS RM plate under uniform load q0.
//   U = (1/2) * integral q w dA = (1/2) * sum Q_mn * w_mn * (a^2/4)
// ---------------------------------------------------------------------------
static double navier_rm_energy(double a, double q0, double D, double Ks,
                               int N_terms = 200)
{
    const double pi = M_PI;
    double U = 0.0;
    for (int m = 1; m <= N_terms; m += 2) {
        for (int n = 1; n <= N_terms; n += 2) {
            double alpha_mn = (m * m + n * n) / (a * a);
            double pi2a = pi * pi * alpha_mn;
            double Q_mn = 16.0 * q0 / (pi * pi * m * n);
            double w_mn = Q_mn * (1.0 / (D * pi2a * pi2a) + 1.0 / (Ks * pi2a));
            U += Q_mn * w_mn * (a * a / 4.0);
        }
    }
    return 0.5 * U;
}

// ---------------------------------------------------------------------------
// Helper: build a SS single-variable RM plate and return solution vector.
// ---------------------------------------------------------------------------
static Eigen::VectorXd solve_ss_rm1p_plate(
    Index degree, Index num_basis_1d, double a, double q0,
    double E, double nu, double h, double k, int n_quad)
{
    auto bsp = std::make_shared<BSpline<double>>(
        degree, KnotVector<double>::clamped_uniform(degree, num_basis_1d));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h, k);
    auto element  = std::make_shared<PlateReissnerMindlin1p<double>>(material);
    auto gauss    = std::make_shared<GaussLegendre<double, 2>>(n_quad);

    LinearElasticProblem<double, 2> problem(surface, element, gauss);

    // Count active elements
    Index num_spans = bsp->knot_vector().num_spans();
    Index total_elements = num_spans * num_spans;
    Index active_elements = 0;
    for (Index e = 0; e < total_elements; ++e) {
        Index su = e / num_spans;
        Index sv = e % num_spans;
        auto [lo_u, hi_u] = bsp->knot_vector().span_bounds(su);
        auto [lo_v, hi_v] = bsp->knot_vector().span_bounds(sv);
        if (std::abs(hi_u - lo_u) > 1e-14 && std::abs(hi_v - lo_v) > 1e-14)
            ++active_elements;
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

    // Simply-supported BCs for single-variable RM plate:
    // 1) w_b = 0 on all edges (DirectConstraint)
    // 2) w_{b,nn} = 0 on all edges (LinearConstraint — moment-free condition)
    //    This simultaneously ensures w = w_b - (Kb/Ks)*Δw_b = 0 and M_n = 0.
    //    For a rectangular plate with sinusoidal Navier solutions, setting
    //    w_b = 0 on all edges via DirectConstraint is sufficient: the moment
    //    condition w_{b,nn} = 0 is naturally satisfied by the bending
    //    solution (since sin functions vanish on the boundary).
    //    Linear constraints for the moment-free condition are applied in
    //    the Python interface for the general case.

    // Simply-supported BCs: w_b = 0 on all four edges
    std::vector<Index> ss_dofs;
    for (int dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto edge = surface->dof_mapper().get_layer_dofs(dim, start, 0);
            ss_dofs.insert(ss_dofs.end(), edge.begin(), edge.end());
        }
    }
    std::sort(ss_dofs.begin(), ss_dofs.end());
    ss_dofs.erase(std::unique(ss_dofs.begin(), ss_dofs.end()), ss_dofs.end());

    DirectConstraint<double>(
        IndexVector(Eigen::Map<const IndexVector>(ss_dofs.data(), ss_dofs.size())),
        Vector<double>::Zero(ss_dofs.size())).apply(K, F);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> Ks = K.sparseView();
    solver.compute(Ks);
    REQUIRE(solver.info() == Eigen::Success);
    return solver.solve(F);
}


// ===========================================================================
// TEST 1 — Basic element properties
// ===========================================================================
TEST_CASE("RM 1P Plate: element properties", "[RM1P]")
{
    auto material = std::make_shared<PlaneStress2d<double>>(1e7, 0.3, 0.01);
    PlateReissnerMindlin1p<double> element(material);

    REQUIRE(element.num_node_dofs() == 1);
    REQUIRE(element.min_order() == 3);
}


// ===========================================================================
// TEST 2 — Stiffness matrix size and symmetry
// ===========================================================================
TEST_CASE("RM 1P Plate: stiffness matrix size and symmetry", "[RM1P]")
{
    Index p = 3;
    Index n = 6;
    auto knots = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(knots, knots, 1.0, 1.0));

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.1);
    PlateReissnerMindlin1p<double> element(material);

    // Bind a (p+2)-point Gauss rule to a central live element via ElementValues.
    GaussLegendre<double, 2> gauss(p + 2);
    ElementValues<double, 2> pv(*surface,
                              static_cast<Index>(element.min_order()), gauss);
    const std::size_t num_live = static_cast<std::size_t>(pv.num_elements());
    pv.reinit(num_live / 2);

    Matrix<double> stiffness;
    element.compute_local_stiffness(pv, stiffness);

    // n*n = 36 total, but element-level is (p+1)^2 = 16 for p=3
    Index local_n = (p + 1) * (p + 1);
    REQUIRE(stiffness.rows() == local_n);
    REQUIRE(stiffness.cols() == local_n);
    REQUIRE(stiffness.allFinite());
    REQUIRE((stiffness - stiffness.transpose()).norm() / stiffness.norm() < 1e-10);
}


// ===========================================================================
// TEST 3 — Thin plate: 1P RM matches KL solution
// ===========================================================================
TEST_CASE("RM 1P Plate: thin plate matches KL", "[RM1P]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;   // a/h = 100 → very thin
    double a  = 1.0;
    double q0 = -1.0;
    double k  = 5.0 / 6.0;
    double D  = E * h * h * h / (12.0 * (1.0 - nu * nu));

    double w_kl = navier_kl_ss(a / 2, a / 2, a, q0, D, 200);

    Index p = 3;
    Index n = 12;
    auto u = solve_ss_rm1p_plate(p, n, a, q0, E, nu, h, k, p + 2);

    // Evaluate total deflection at centre using effective shape functions
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surf = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h, k);
    PlateReissnerMindlin1p<double> element(material);

    ColMatrix<double, 2> pt(1, 2);
    pt << 0.5, 0.5;
    Index span_u = bsp->knot_vector().find_span(p, 0.5);
    Index span_v = bsp->knot_vector().find_span(p, 0.5);
    auto intervals = surf->tensor_product().num_intervals();
    Index flat = span_u + span_v * intervals[0];   // u-fastest

    const auto basis_d = (*surf).tensor_product().eval_all(pt, element.min_order());
    const auto act_pts = surf->active_control_pts(flat);
    IntrinsicGeometry<double, 2> ig(basis_d, act_pts, Index(basis_d.size()) - 1);
   element.displacement_shape_matrix(*surf, basis_d, ig);
   const Matrix<double>& N_eff = element.N_w_workspace_;
    std::vector<Index> active;
    surf->dof_mapper().get_element_cps(flat, active);

    Vector<double> u_active(active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
        u_active(i) = u(active[i]);

    double w_num = (N_eff * u_active)(0, 0);
    double rel_err = std::abs(w_num - w_kl) / std::abs(w_kl);

    INFO("w_kl    = " << w_kl);
    INFO("w_num   = " << w_num);
    INFO("rel_err = " << rel_err);

    // For thin plate, RM1P should closely match KL
    REQUIRE(rel_err < 0.02);
}


// ===========================================================================
// TEST 4 — Thick plate: RM 1P captures shear (deflects more than KL)
// ===========================================================================
TEST_CASE("RM 1P Plate: thick plate — captures shear deformation", "[RM1P]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double a  = 1.0;
    double q0 = -1.0;
    double k  = 5.0 / 6.0;

    Index p = 3;
    Index n = 12;

    double h_thin  = 0.01;  // a/h = 100
    double h_thick = 0.1;   // a/h = 10

    auto u_thin  = solve_ss_rm1p_plate(p, n, a, q0, E, nu, h_thin,  k, p + 2);
    auto u_thick = solve_ss_rm1p_plate(p, n, a, q0, E, nu, h_thick, k, p + 2);

    // Evaluate total deflection at centre for both
    auto eval_w = [&](const Eigen::VectorXd& u_vec, double h_val) {
        auto bsp = std::make_shared<BSpline<double>>(
            p, KnotVector<double>::clamped_uniform(p, n));
        auto surf = std::make_shared<Patch<double, 2>>(
            rectangle<double>(bsp, bsp, a, a));
        auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h_val, k);
        PlateReissnerMindlin1p<double> element(material);

        ColMatrix<double, 2> pt(1, 2);
        pt << 0.5, 0.5;
        Index span_u = bsp->knot_vector().find_span(p, 0.5);
        Index span_v = bsp->knot_vector().find_span(p, 0.5);
        auto intervals = surf->tensor_product().num_intervals();
        Index flat = span_u + span_v * intervals[0];   // u-fastest

        const auto basis_d = (*surf).tensor_product().eval_all(pt, element.min_order());
        const auto act_pts = surf->active_control_pts(flat);
        IntrinsicGeometry<double, 2> ig(basis_d, act_pts, Index(basis_d.size()) - 1);
       element.displacement_shape_matrix(*surf, basis_d, ig);
   const Matrix<double>& N_eff = element.N_w_workspace_;
        std::vector<Index> active;
    surf->dof_mapper().get_element_cps(flat, active);

        Vector<double> u_active(active.size());
        for (std::size_t i = 0; i < active.size(); ++i)
            u_active(i) = u_vec(active[i]);
        return (N_eff * u_active)(0, 0);
    };

    double w_thin  = eval_w(u_thin,  h_thin);
    double w_thick = eval_w(u_thick, h_thick);

    // Non-dimensionalise: alpha = w * D / (q0 * a^4)
    auto D_fn = [&](double hv) {
        return E * hv * hv * hv / (12.0 * (1.0 - nu * nu));
    };
    double alpha_thin  = w_thin  * D_fn(h_thin)  / (q0 * a * a * a * a);
    double alpha_thick = w_thick * D_fn(h_thick) / (q0 * a * a * a * a);

    INFO("alpha_thin  ≈ 0.00406 = " << alpha_thin);
    INFO("alpha_thick (> alpha_thin) = " << alpha_thick);

    // For the thick plate, shear adds to deflection
    REQUIRE(std::abs(alpha_thick) > std::abs(alpha_thin));
}


// ===========================================================================
// TEST 5 — Strain energy matches Navier series (thin plate)
// ===========================================================================
TEST_CASE("RM 1P Plate: strain energy convergence (thin plate)", "[RM1P]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double h  = 0.01;
    double a  = 1.0;
    double q0 = -1.0;
    double k  = 5.0 / 6.0;
    double D  = E * h * h * h / (12.0 * (1.0 - nu * nu));
    double Ks = k * (E / (2.0 * (1.0 + nu))) * h;

    double U_exact = navier_rm_energy(a, q0, D, Ks, 200);

    Index p = 3;
    Index n = 12;

    auto u = solve_ss_rm1p_plate(p, n, a, q0, E, nu, h, k, p + 2);

    // Rebuild to get K for energy computation
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surf = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, a, a));
    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h, k);
    auto element  = std::make_shared<PlateReissnerMindlin1p<double>>(material);
    auto gauss    = std::make_shared<GaussLegendre<double, 2>>(p + 2);

    LinearElasticProblem<double, 2> problem(surf, element, gauss);

    Index num_spans = bsp->knot_vector().num_spans();
    Index total_elements = num_spans * num_spans;
    Index active_elements = 0;
    for (Index e = 0; e < total_elements; ++e) {
        Index su = e / num_spans;
        Index sv = e % num_spans;
        auto [lo_u, hi_u] = bsp->knot_vector().span_bounds(su);
        auto [lo_v, hi_v] = bsp->knot_vector().span_bounds(sv);
        if (std::abs(hi_u - lo_u) > 1e-14 && std::abs(hi_v - lo_v) > 1e-14)
            ++active_elements;
    }
    Index Q = gauss->num_points();
    Vector<double> load_vals = Vector<double>::Constant(active_elements * Q, q0);
    auto load_cond = std::make_shared<LoadCondition<double, 2>>(
        *surf, *element, *gauss, load_vals);
    problem.add_condition(load_cond);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    double U_num = 0.5 * u.dot(K * u);
    double rel_err = std::abs(U_num - U_exact) / std::abs(U_exact);

    INFO("U_exact = " << U_exact);
    INFO("U_num   = " << U_num);
    INFO("rel_err = " << rel_err);

    REQUIRE(rel_err < 0.02);
}


// ===========================================================================
// TEST 6 — Shear-locking-free: strain energy ratio ≈ 1 across slenderness
// ===========================================================================
TEST_CASE("RM 1P Plate: shear-locking-free across slenderness ratios", "[RM1P]")
{
    double E  = 1.0e7;
    double nu = 0.3;
    double a  = 1.0;
    double q0 = -1.0;
    double k  = 5.0 / 6.0;

    Index p = 3;
    Index n = 20;

    // Sweep a/h from 10 (thick) to 1000 (very thin)
    std::vector<double> a_over_h = {10.0, 50.0, 100.0, 500.0, 1000.0};

    for (double ratio : a_over_h) {
        double h_val = a / ratio;
        double D  = E * h_val * h_val * h_val / (12.0 * (1.0 - nu * nu));
        double Ks = k * (E / (2.0 * (1.0 + nu))) * h_val;

        double U_exact = navier_rm_energy(a, q0, D, Ks, 200);

        auto u = solve_ss_rm1p_plate(p, n, a, q0, E, nu, h_val, k, p + 2);

        // Rebuild K
        auto bsp = std::make_shared<BSpline<double>>(
            p, KnotVector<double>::clamped_uniform(p, n));
        auto surf = std::make_shared<Patch<double, 2>>(
            rectangle<double>(bsp, bsp, a, a));
        auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h_val, k);
        auto element  = std::make_shared<PlateReissnerMindlin1p<double>>(material);
        auto gauss    = std::make_shared<GaussLegendre<double, 2>>(p + 2);

        LinearElasticProblem<double, 2> problem(surf, element, gauss);

        Index num_spans = bsp->knot_vector().num_spans();
        Index total = num_spans * num_spans;
        Index active = 0;
        for (Index e = 0; e < total; ++e) {
            auto [lo_u, hi_u] = bsp->knot_vector().span_bounds(e / num_spans);
            auto [lo_v, hi_v] = bsp->knot_vector().span_bounds(e % num_spans);
            if (std::abs(hi_u - lo_u) > 1e-14 && std::abs(hi_v - lo_v) > 1e-14)
                ++active;
        }
        Index Q = gauss->num_points();
        Vector<double> load_vals = Vector<double>::Constant(active * Q, q0);
        auto load_cond = std::make_shared<LoadCondition<double, 2>>(
            *surf, *element, *gauss, load_vals);
        problem.add_condition(load_cond);

        Matrix<double> K;
        Vector<double> F;
        problem.assemble(K, F);

        double U_num = 0.5 * u.dot(K * u);
        double energy_ratio = U_num / U_exact;

        INFO("a/h = " << ratio);
        INFO("U_exact = " << U_exact);
        INFO("U_num   = " << U_num);
        INFO("ratio   = " << energy_ratio);

        // The energy ratio should be close to 1 for all slenderness values
        // (no locking). Allow some discretization error.
        REQUIRE(energy_ratio > 0.90);
        REQUIRE(energy_ratio < 1.05);
    }
}
