#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "boundary_field.hpp"
#include "patch_boundary.hpp"
#include "physical_points.hpp"
#include "bspline.hpp"
#include "direct_constraint.hpp"
#include "factories.hpp"
#include "gauss_legendre.hpp"
#include "lagrange_boundary_condition.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "patch.hpp"
#include "plane_stress_2d.hpp"
#include "plate_reissner_mindlin_1p.hpp"
#include "plate_reissner_mindlin_3p.hpp"

using namespace pyck;

template <typename T>
static void add_clamped_lagrange_all_edges(
    LinearElasticProblem<T, 2>& problem,
    std::vector<Ptr<PatchBoundary<T, 2>>>& boundaries,
    const Ptr<Patch<T, 2>>& surface,
    const Element<T, 2>& element,
    const QuadratureRule<T, 1>& gauss1d)
{
    for (std::size_t dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto bdy = create_patch_boundary<T, 2>(surface, dim, start);
            boundaries.push_back(bdy);
            auto cond = std::make_shared<LagrangeBoundaryCondition<T, 2>>(
                *boundaries.back(), element, gauss1d);
            cond->add(std::make_shared<TransverseDisplacement<T>>(), T(0));
            cond->add(std::make_shared<NormalRotation<T>>(), T(0));
            cond->add(std::make_shared<TangentialRotation<T>>(), T(0));
            problem.add_condition(cond);
        }
    }
}

template <typename T>
static void add_clamped_direct_rm3(
    LinearElasticProblem<T, 2>& problem,
    const Ptr<Patch<T, 2>>& surface)
{
    std::vector<Index> fixed;
    for (std::size_t dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto layer = surface->layer_dofs(dim, start, 0);
            for (auto node : layer) {
                fixed.push_back(3 * node);
                fixed.push_back(3 * node + 1);
                fixed.push_back(3 * node + 2);
            }
        }
    }

    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());
    problem.add_direct_constraint(
        std::make_shared<DirectConstraint<T>>(
            IndexVector(Eigen::Map<const IndexVector>(fixed.data(), fixed.size())),
            T(0))
    );
}

static Vector<double> make_uniform_load(
    const Ptr<Patch<double, 2>>& surface,
    const QuadratureRule<double, 2>& gauss2d,
    double q0)
{
    auto phys_pts = pyck::eval_physical_points(*surface, gauss2d);
    Vector<double> load_vals(phys_pts.rows());
    load_vals.setConstant(q0);
    return load_vals;
}

TEST_CASE("LagrangeBoundaryCondition augments the system symmetrically", "[conditions][lagrange]")
{
    Index p = 2;
    Index n = 5;
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, 1.0, 1.0));

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e7, 0.3, 0.1);
    auto element = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss2d = std::make_shared<GaussLegendre<double, 2>>(p + 1);
    GaussLegendre<double, 1> gauss1d(p + 1);

    auto boundary = create_patch_boundary<double, 2>(surface, 0, true);
    auto cond = std::make_shared<LagrangeBoundaryCondition<double, 2>>(
        *boundary, *element, gauss1d);
    cond->add(std::make_shared<TransverseDisplacement<double>>(), 0.0);
    cond->add(std::make_shared<NormalRotation<double>>(), 0.0);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);
    problem.add_condition(cond);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    const Index n_primal = static_cast<Index>(surface->num_control_pts()) * 3;
    const Index n_lambda = static_cast<Index>(boundary->num_control_pts()) * 2;

    REQUIRE(K.rows() == n_primal + n_lambda);
    REQUIRE(K.cols() == n_primal + n_lambda);
    REQUIRE(F.size() == n_primal + n_lambda);
    REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());
    REQUIRE(F.norm() == Approx(0.0).margin(1e-14));
}

TEST_CASE("LagrangeBoundaryCondition matches direct clamped RM3 solution", "[conditions][lagrange]")
{
    const double E = 1.0e7;
    const double nu = 0.3;
    const double h = 0.1;
    const double q0 = -1.0;
    const Index p = 3;
    const Index n = 9;

    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, 1.0, 1.0));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss2d = std::make_shared<GaussLegendre<double, 2>>(p + 1);
    GaussLegendre<double, 1> gauss1d(p + 1);
    auto load = make_uniform_load(surface, *gauss2d, q0);

    LinearElasticProblem<double, 2> lm_problem(surface, element, gauss2d);
    lm_problem.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *surface, *element, *gauss2d, load));
    std::vector<Ptr<PatchBoundary<double, 2>>> lm_boundaries;
    add_clamped_lagrange_all_edges(
        lm_problem, lm_boundaries, surface, *element, gauss1d);

    LinearElasticProblem<double, 2> direct_problem(surface, element, gauss2d);
    direct_problem.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *surface, *element, *gauss2d, load));
    add_clamped_direct_rm3(direct_problem, surface);

    Matrix<double> K_lm, K_dir;
    Vector<double> F_lm, F_dir;
    lm_problem.assemble(K_lm, F_lm);
    direct_problem.assemble(K_dir, F_dir);

    Eigen::PartialPivLU<Matrix<double>> lm_solver(K_lm);
    Eigen::PartialPivLU<Matrix<double>> dir_solver(K_dir);
    REQUIRE(lm_solver.info() == Eigen::Success);
    REQUIRE(dir_solver.info() == Eigen::Success);

    Vector<double> u_lm = lm_solver.solve(F_lm);
    Vector<double> u_dir = dir_solver.solve(F_dir);
    REQUIRE(lm_solver.info() == Eigen::Success);
    REQUIRE(dir_solver.info() == Eigen::Success);

    const Index n_primal = static_cast<Index>(surface->num_control_pts()) * 3;
    const Vector<double> diff = u_lm.head(n_primal) - u_dir;
    REQUIRE(u_dir.norm() > 0.0);
    const double rel = diff.norm() / u_dir.norm();
    REQUIRE(rel < 1e-10);
}

TEST_CASE("LagrangeBoundaryCondition solves an RM1 simply-supported plate", "[conditions][lagrange]")
{
    const double E = 1.0e7;
    const double nu = 0.3;
    const double h = 0.05;
    const double f0 = -10.0;
    const double L = 1.0;
    const double W = 1.0;
    const Index p = 3;
    const Index n = 8;

    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, L, W));

    auto material = std::make_shared<PlaneStress2d<double>>(E, nu, h);
    auto element = std::make_shared<PlateReissnerMindlin1p<double>>(material);
    auto gauss2d = std::make_shared<GaussLegendre<double, 2>>(p + 1);
    GaussLegendre<double, 1> gauss1d(p + 1);

    auto phys_pts = pyck::eval_physical_points(*surface, *gauss2d);
    Vector<double> load_vals(phys_pts.rows());
    for (Index i = 0; i < phys_pts.rows(); ++i) {
        load_vals(i) = f0 * std::sin(M_PI * phys_pts(i, 0) / L)
                         * std::sin(M_PI * phys_pts(i, 1) / W);
    }

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);
    problem.add_condition(std::make_shared<LoadCondition<double, 2>>(
        *surface, *element, *gauss2d, load_vals));

    std::vector<Ptr<PatchBoundary<double, 2>>> lm_boundaries;
    for (std::size_t dim = 0; dim < 2; ++dim) {
        for (bool start : {true, false}) {
            auto bdy = create_patch_boundary<double, 2>(surface, dim, start);
            lm_boundaries.push_back(bdy);
            auto cond = std::make_shared<LagrangeBoundaryCondition<double, 2>>(
                *lm_boundaries.back(), *element, gauss1d);
            cond->add(std::make_shared<TransverseDisplacement<double>>(), 0.0);
            cond->add(std::make_shared<TangentialRotation<double>>(), 0.0);
            problem.add_condition(cond);
        }
    }

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    Eigen::PartialPivLU<Matrix<double>> solver(K);
    REQUIRE(solver.info() == Eigen::Success);

    Vector<double> u = solver.solve(F);
    REQUIRE(solver.info() == Eigen::Success);

    const Index n_primal = static_cast<Index>(surface->num_control_pts());
    const double max_abs_u = u.head(n_primal).lpNorm<Eigen::Infinity>();
    REQUIRE(std::isfinite(max_abs_u));
    REQUIRE(max_abs_u > 0.0);
}
