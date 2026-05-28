#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "boundary_field.hpp"
#include "patch_boundary.hpp"
#include "bspline.hpp"
#include "direct_constraint.hpp"
#include "factories.hpp"
#include "gauss_legendre.hpp"
#include "boundary_lagrange_condition.hpp"
#include "linear_elastic_problem.hpp"
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

    SparseMatrix<double> K;
    Vector<double> F;
    problem.assemble(K, F);

    const Index n_primal = static_cast<Index>(surface->num_control_pts()) * 3;
    const Index n_lambda = static_cast<Index>(boundary->num_control_pts()) * 2;

    REQUIRE(K.rows() == n_primal + n_lambda);
    REQUIRE(K.cols() == n_primal + n_lambda);
    REQUIRE(F.size() == n_primal + n_lambda);
    REQUIRE((K - SparseMatrix<double>(K.transpose())).norm() < 1e-12 * K.norm());
    REQUIRE(F.norm() == Approx(0.0).margin(1e-14));
}

