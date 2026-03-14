#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#include "direct_constraint.hpp"
#include "linear_constraint.hpp"

using namespace pyck;

TEST_CASE("DirectConstraint enforces prescribed DOF values", "[constraints]") 
{
    Matrix<double> K(3, 3);
    K << 2, -1, 0,
        -1, 2, -1,
         0, -1, 1;
    Vector<double> F(3);
    F << 1, 2, 3;

    DirectConstraint<double> dirichlet_cond({0, 2}, std::vector<double>{10.0, -5.0});
    dirichlet_cond.apply(K, F);

    // Verify diagonal entries and RHS values for constrained DOFs
    REQUIRE(K(0, 0) == 1.0);
    REQUIRE(K(0, 1) == 0.0);
    REQUIRE(K(0, 2) == 0.0);
    REQUIRE(F(0) == 10.0);
 
    REQUIRE(K(2, 0) == 0.0);
    REQUIRE(K(2, 1) == 0.0);
    REQUIRE(K(2, 2) == 1.0);
    REQUIRE(F(2) == -5.0);
 
    // Verify off-diagonal entries in constrained rows/columns are zeroed 
    // and condensed into load vector F
    REQUIRE(K(1, 0) == 0.0);
    REQUIRE(K(1, 2) == 0.0);
    REQUIRE(F(1) == 7.0);
}

TEST_CASE("LinearConstraint enforces identity relationships between DOFs", "[constraints]") 
{
    Matrix<double> K(3, 3);
    K << 2, -1, -1,
        -1, 3, -2,
        -1, -2, 4;
    Vector<double> F(3);
    F << 10, 20, 30;

    IndexMatrix ms_masters(1, 1);
    ms_masters << 0;
    LinearConstraint<double> ms_cond({2}, ms_masters, {1.0}); // slave 2, master 0
    ms_cond.apply(K, F);

    // Verify symmetry of condensed stiffness matrix
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            REQUIRE(K(i, j) == Approx(K(j, i)).margin(1e-12));
        }
    }
 
    // Solve constrained linear system
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K.sparseView());
    Vector<double> u = solver.solve(F);
 
    // Verify enforcement of master-slave identity relation (u[2] == u[0])
    REQUIRE(u(2) == Approx(u(0)).margin(1e-12));
}

TEST_CASE("LinearConstraint enforces linear combinations of DOFs", "[constraints]") 
{
    Matrix<double> K(4, 4);
    K << 2, -1,  0, -1,
        -1,  3, -2,  0,
         0, -2,  4, -1,
        -1,  0, -1,  5;
    Vector<double> F(4);
    F << 10, 20, 30, 40;

    // Enforce u_3 = 0.5 * u_0 + (-0.3) * u_1 + 2.0
    // slave = 3, masters = (0, 0.5), (1, -0.3), c = 2.0
    IndexMatrix masters(1, 2);
    masters << 0, 1;
    LinearConstraint<double> mpc_cond({3}, masters, {0.5, -0.3}, 2.0);
    mpc_cond.apply(K, F);

    // Verify symmetry of condensed stiffness matrix
    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) {
            REQUIRE(K(i, j) == Approx(K(j, i)).margin(1e-12));
        }
    }
 
    // Solve system and verify constraint equation holds for solution:
    // u[3] = 0.5 * u[0] - 0.3 * u[1] + 2.0
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K.sparseView());
    Vector<double> u = solver.solve(F);
 
    REQUIRE(u(3) == Approx(0.5 * u(0) - 0.3 * u(1) + 2.0).margin(1e-12));
}
