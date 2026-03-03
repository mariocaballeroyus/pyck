#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include "knots.hpp"
#include "bspline.hpp"
#include "curve.hpp"
#include "gauss_legendre.hpp"
#include "load_condition.hpp"

using namespace pyck;

TEST_CASE("LoadCondition 1D", "[conditions][load_condition]") {
    
    // 1. Create a simple 1D curve (line segment from x=0 to x=2)
    std::vector<double> knots_vec = {0.0, 0.0, 1.0, 2.0, 2.0};
    auto basis = std::make_shared<BSpline<double>>(1, KnotVector<double>(knots_vec));
    
    Index num_pts = basis->num_basis(); // 3 control points
    ColMatrix<double, 3> control_pts(num_pts, 3);
    control_pts.setZero();
    control_pts(0, 0) = 0.0;
    control_pts(1, 0) = 1.0;
    control_pts(2, 0) = 2.0;
    
    CurvePatch<double> curve(basis, control_pts);

    // 2. Create a quadrature rule (Gauss-Legendre, 2 points per element should be enough for linear shape funcs)
    GaussLegendre<double> quad(2);

    SECTION("Constant Body Force") {
        // t(x) = 10.0, with 2 non-zero elements and 2 quad points each = 4 total
        Vector<double> load_values = Vector<double>::Constant(4, 10.0);
        
        LoadCondition<double, 1> load_cond(curve, quad, load_values);
        
        Vector<double> F(num_pts);
        F.setZero();
        Matrix<double> K(num_pts, num_pts); // dummy
        
        load_cond.apply(K, F);
        
        // Total load should be \int_0^2 10 dx = 20.
        // Distributed among 3 nodes based on shape functions.
        // For linear basis on uniform knots:
        // F_0 = \int N_0 * 10 = 10 * (1/2 * 1) = 5
        // F_1 = \int N_1 * 10 = 10 * (1) = 10
        // F_2 = \int N_2 * 10 = 10 * (1/2 * 1) = 5
        REQUIRE(F(0) == Approx(5.0));
        REQUIRE(F(1) == Approx(10.0));
        REQUIRE(F(2) == Approx(5.0));
        REQUIRE(F.sum() == Approx(20.0));
    }
}
