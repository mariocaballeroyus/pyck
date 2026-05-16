#include "catch.hpp"
#include "uniaxial_stress_1d.hpp"
#include "plane_stress_2d.hpp"

using namespace pyck;

TEST_CASE("UniaxialStress1d", "[materials]") {
    double E = 210e9;
    double A = 0.01;
    double I = 0.0001;
    double nu = 0.3;
    double G = E / (2.0 * (1.0 + nu));
    double k = 5.0 / 6.0;

    UniaxialStress1d<double> mat(E, nu, A, I, k);

    REQUIRE(mat.youngs_modulus() == Approx(E));
    REQUIRE(mat.shear_modulus() == Approx(G));
    REQUIRE(mat.section_area() == Approx(A));
    REQUIRE(mat.moment_inertia_22() == Approx(I));
    REQUIRE(mat.shear_coefficient_22() == Approx(k));
    CHECK(mat.name() == "UniaxialStress1d");
}

TEST_CASE("PlaneStress2d", "[materials]") {
    double E = 210e9;
    double nu = 0.3;
    double h = 0.05;

    PlaneStress2d<double> mat(E, nu, h);

    REQUIRE(mat.youngs_modulus() == Approx(E));
    REQUIRE(mat.poisson_ratio() == Approx(nu));
    REQUIRE(mat.thickness() == Approx(h));
    CHECK(mat.name() == "PlaneStress2d");

    auto C = mat.constitutive_matrix();
    double factor = E / (1.0 - nu * nu);
    REQUIRE(C(0, 0) == Approx(factor));
    REQUIRE(C(0, 1) == Approx(factor * nu));
    REQUIRE(C(2, 2) == Approx(E / (2.0 * (1.0 + nu))));
}
