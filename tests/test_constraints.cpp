#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "strong_dirichlet_constraint.hpp"
#include "master_slave_constraint.hpp"

using namespace pyck;

TEST_CASE("StrongDirichletConstraint forces values", "[constraints]") {
    Matrix<double> K(3, 3);
    K << 2, -1, 0,
        -1, 2, -1,
         0, -1, 1;
    Vector<double> F(3);
    F << 1, 2, 3;

    StrongDirichletConstraint<double> dirichlet_cond({0, 2}, std::vector<double>{10.0, -5.0});
    dirichlet_cond.apply(K, F);

    // After assign_scalar:
    // Rows and columns 0 and 2 are identity logic
    REQUIRE(K(0, 0) == 1.0);
    REQUIRE(K(0, 1) == 0.0);
    REQUIRE(K(0, 2) == 0.0);
    REQUIRE(F(0) == 10.0);

    REQUIRE(K(2, 0) == 0.0);
    REQUIRE(K(2, 1) == 0.0);
    REQUIRE(K(2, 2) == 1.0);
    REQUIRE(F(2) == -5.0);

    // Col 0 and 2 on row 1 should be 0, and F(1) should have effects subtracted
    REQUIRE(K(1, 0) == 0.0);
    REQUIRE(K(1, 2) == 0.0);
    // original F(1) was 2. Col 0 originally had -1, col 2 had -1.
    // F(1) gets modified: F(1) -= K_orig(1, 0)*10.0 + K_orig(1, 2)*(-5.0)
    // F(1) -= (-1)*10.0 + (-1)*(-5.0)
    // F(1) -= -10 + 5 
    // F(1) -= -5 => F(1) = 2 - (-5) = 7
    REQUIRE(F(1) == 7.0);
}

TEST_CASE("MasterSlaveConstraint enforces equality and preserves symmetry", "[constraints]") {
    Matrix<double> K(3, 3);
    K << 2, -1, -1,
        -1, 3, -2,
        -1, -2, 4;
    Vector<double> F(3);
    F << 10, 20, 30;

    MasterSlaveConstraint<double> ms_cond({{0, 2}}); // slave 2, master 0
    ms_cond.apply(K, F);

    // K_orig:
    // 2  -1  -1
    // -1  3  -2
    // -1 -2   4
    
    // Check symmetry
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            REQUIRE(K(i, j) == Approx(K(j, i)).margin(1e-12));
        }
    }

    // Solve
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K.sparseView());
    Vector<double> u = solver.solve(F);

    // slave 2 = master 0
    REQUIRE(u(2) == Approx(u(0)).margin(1e-12));
}
