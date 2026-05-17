#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "bspline.hpp"
#include "factories.hpp"
#include "gauss_legendre.hpp"
#include "lagrange_domain_condition.hpp"
#include "linear_elastic_problem.hpp"
#include "load_condition.hpp"
#include "patch.hpp"
#include "plane_stress_2d.hpp"
#include "plate_reissner_mindlin_3p.hpp"

using namespace pyck;

TEST_CASE("LagrangeDomainCondition augments the system symmetrically",
          "[conditions][lagrange_domain]")
{
    const Index p = 2;
    const Index n = 5;
    const double L = 2.0;
    const double W = 3.0;
    const double area = L * W;

    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, L, W));

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e7, 0.3, 0.1);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);

    const double g_bar = 0.7;
    auto cond = std::make_shared<LagrangeDomainCondition<double, 2>>(
        *surface, *element, *gauss2d);
    cond->add(/*dof_index=*/0, g_bar);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);
    problem.add_condition(cond);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    const Index n_cps    = static_cast<Index>(surface->num_control_pts());
    const Index n_primal = n_cps * 3;
    const Index n_lambda = 1;

    REQUIRE(K.rows() == n_primal + n_lambda);
    REQUIRE(K.cols() == n_primal + n_lambda);
    REQUIRE(F.size() == n_primal + n_lambda);
    REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());

    const Index lambda_row = n_primal;

    // RHS = value · |Ω|.
    REQUIRE(F(lambda_row) == Approx(g_bar * area).margin(1e-12));

    // Constraint row applied to the constant w-mode reproduces |Ω|
    // (basis partition-of-unity).
    Vector<double> ones_w = Vector<double>::Zero(n_primal);
    for (Index i = 0; i < n_cps; ++i) ones_w(3 * i + 0) = 1.0;
    const double dot_w = K.row(lambda_row).head(n_primal).dot(ones_w);
    REQUIRE(dot_w == Approx(area).margin(1e-10));

    // Other slots (theta_x, theta_y) carry zero coefficients in the
    // constraint row.
    for (Index i = 0; i < n_cps; ++i) {
        REQUIRE(K(lambda_row, 3 * i + 1) == Approx(0.0).margin(1e-14));
        REQUIRE(K(lambda_row, 3 * i + 2) == Approx(0.0).margin(1e-14));
    }
}

TEST_CASE("LagrangeDomainCondition supports multiple slots",
          "[conditions][lagrange_domain]")
{
    const Index p = 2;
    const Index n = 4;
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, 1.0, 1.0));   // area = 1

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e7, 0.3, 0.1);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);

    auto cond = std::make_shared<LagrangeDomainCondition<double, 2>>(
        *surface, *element, *gauss2d);
    cond->add(0, 1.0);
    cond->add(1, 2.0);

    LinearElasticProblem<double, 2> problem(surface, element, gauss2d);
    problem.add_condition(cond);

    Matrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    const Index n_primal = static_cast<Index>(surface->num_control_pts()) * 3;
    REQUIRE(K.rows() == n_primal + 2);
    REQUIRE((K - K.transpose()).norm() < 1e-12 * K.norm());
    REQUIRE(F(n_primal + 0) == Approx(1.0).margin(1e-12));
    REQUIRE(F(n_primal + 1) == Approx(2.0).margin(1e-12));

    // The two constraint rows act on disjoint DOF slots.
    const Index n_cps = static_cast<Index>(surface->num_control_pts());
    for (Index i = 0; i < n_cps; ++i) {
        REQUIRE(K(n_primal + 0, 3 * i + 1) == Approx(0.0).margin(1e-14));
        REQUIRE(K(n_primal + 0, 3 * i + 2) == Approx(0.0).margin(1e-14));
        REQUIRE(K(n_primal + 1, 3 * i + 0) == Approx(0.0).margin(1e-14));
        REQUIRE(K(n_primal + 1, 3 * i + 2) == Approx(0.0).margin(1e-14));
    }
}

TEST_CASE("LagrangeDomainCondition rejects out-of-range dof_index",
          "[conditions][lagrange_domain]")
{
    const Index p = 2;
    const Index n = 4;
    auto bsp = std::make_shared<BSpline<double>>(
        p, KnotVector<double>::clamped_uniform(p, n));
    auto surface = std::make_shared<Patch<double, 2>>(
        rectangle<double>(bsp, bsp, 1.0, 1.0));

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e7, 0.3, 0.1);
    auto element  = std::make_shared<PlateReissnerMindlin3p<double>>(material);
    auto gauss2d  = std::make_shared<GaussLegendre<double, 2>>(p + 1);

    LagrangeDomainCondition<double, 2> cond(*surface, *element, *gauss2d);
    REQUIRE_THROWS_AS(cond.add(3, 0.0), std::invalid_argument);
}
