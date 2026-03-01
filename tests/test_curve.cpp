#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>
#include <string>

#include "bspline.hpp"
#include "knots.hpp"
#include "curve.hpp"

using namespace pyck;

TEST_CASE("CurvePatch: Analytical Push-Forward Verification", "[geometry][curve]") {
    std::vector<double> knots_vec = {0, 0, 0, 0, 1, 1, 1, 1};
    auto basis = std::make_shared<BSpline<double>>(3, KnotVector<double>(knots_vec));

    Eigen::MatrixXd control_pts(4, 3);
    control_pts.row(0) << 0.0,  0.0, 0.0;
    control_pts.row(1) << 1.0,  2.0, 0.0;
    control_pts.row(2) << 2.0, -1.0, 1.0;
    control_pts.row(3) << 3.0,  0.0, 0.0;

    CurvePatch<double> curve(basis, control_pts);

    std::vector<double> test_pts = {0.15, 0.5, 0.85};

    for (double u : test_pts)
    {
        SECTION("Evaluation at u = " + std::to_string(u)) {
            Eigen::MatrixXd params(1, 1);
            params << u;

            std::vector<double> N_u = {
                -3.0 * std::pow(1.0 - u, 2),
                 3.0 * (1.0 - u) * (1.0 - 3.0 * u),
                 3.0 * u * (2.0 - 3.0 * u),
                 3.0 * std::pow(u, 2)
            };
            std::vector<double> N_uu = {
                 6.0 * (1.0 - u),
                -6.0 * (2.0 - 3.0 * u),
                 6.0 * (1.0 - 3.0 * u),
                 6.0 * u
            };
            std::vector<double> N_uuu = {-6.0, 18.0, -18.0, 6.0};

            Eigen::Vector3d x_1(3.0, 27*u*u - 30*u + 6, -9*u*u + 6*u);
            Eigen::Vector3d x_11(0.0, 54*u - 30, -18*u + 6);
            Eigen::Vector3d x_111(0.0, 54.0, -18.0);

            double g11 = x_1.dot(x_1);
            double Gamma = x_1.dot(x_11) / g11;
            double Gamma_u = (x_11.dot(x_11) + x_1.dot(x_111)) / g11 - 2.0 * std::pow(Gamma, 2);

            auto r = curve.eval_shape_functions(params, 3);
            
            const auto& dN  = r[1];
            const auto& d2N = r[2];
            const auto& d3N = r[3];

            for (int a = 0; a < 4; ++a) {
                double expected_dN = N_u[a] / std::sqrt(g11);
                CHECK(dN(0, a) == Approx(expected_dN).margin(1e-10));

                double expected_d2N = (N_uu[a] - Gamma * N_u[a]) / g11;
                CHECK(d2N(0, a) == Approx(expected_d2N).margin(1e-10));

                double expected_d3N = (N_uuu[a] - 3.0 * Gamma * N_uu[a] + 
                                      (2.0 * std::pow(Gamma, 2) - Gamma_u) * N_u[a]) / std::pow(g11, 1.5);
                CHECK(d3N(0, a) == Approx(expected_d3N).margin(1e-10));
            }
            
            CHECK(dN.row(0).sum()  == Approx(0.0).margin(1e-11));
            CHECK(d2N.row(0).sum() == Approx(0.0).margin(1e-11));
            CHECK(d3N.row(0).sum() == Approx(0.0).margin(1e-11));
        }
    }
}

TEST_CASE("CurvePatch: External AD Numerical Validation", "[geometry][curve]") {
    std::vector<double> knots_vec = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
    auto basis = std::make_shared<BSpline<double>>(3, KnotVector<double>(knots_vec));

    Eigen::MatrixXd control_pts(4, 3);
    control_pts.row(0) << 0.0,  0.0, 0.0;
    control_pts.row(1) << 1.0,  2.0, 0.0;
    control_pts.row(2) << 2.0, -1.0, 1.0;
    control_pts.row(3) << 3.0,  0.0, 0.0;

    CurvePatch<double> curve(basis, control_pts);

    SECTION("Evaluation at u = 0.15") {
        Eigen::MatrixXd u(1, 1); u << 0.15;
        auto results = curve.eval_shape_functions(u, 3);

        Eigen::Vector4d expected_N(0.6141250000000000, 0.3251250000000000, 0.0573750000000000, 0.0033750000000000);
        Eigen::Vector4d expected_dN(-0.5807828087105747, 0.3758006409303719, 0.1868955059172438, 0.0180866618629591);
        Eigen::Vector4d expected_d2N(-0.1238056625562066, -0.3506754246294261, 0.3946046702878425, 0.0798764168977904);
        Eigen::Vector4d expected_d3N(0.9294862062463560, -1.4201195886353655, 0.2153416673286355, 0.2752917150603740);

        for(int i=0; i<4; ++i) {
            CHECK(results[0](0, i) == Approx(expected_N(i)).margin(1e-12));
            CHECK(results[1](0, i) == Approx(expected_dN(i)).margin(1e-12));
            CHECK(results[2](0, i) == Approx(expected_d2N(i)).margin(1e-12));
            CHECK(results[3](0, i) == Approx(expected_d3N(i)).margin(1e-12));
        }
    }

    SECTION("Evaluation at u = 0.5") {
        Eigen::MatrixXd u(1, 1); u << 0.5;
        auto results = curve.eval_shape_functions(u, 3);

        Eigen::Vector4d expected_N(0.1250000000000000, 0.3750000000000000, 0.3750000000000000, 0.1250000000000000);
        Eigen::Vector4d expected_dN(-0.1961161351381840, -0.1961161351381840, 0.1961161351381840, 0.1961161351381840);
        Eigen::Vector4d expected_d2N(0.2209072978303747, -0.1893491124260355, -0.2209072978303747, 0.1893491124260355);
        Eigen::Vector4d expected_d3N(-0.2691451698330937, 0.2589887483299580, -0.1599636386743859, 0.1701200601775215);

        for(int i=0; i<4; ++i) {
            CHECK(results[0](0, i) == Approx(expected_N(i)).margin(1e-12));
            CHECK(results[1](0, i) == Approx(expected_dN(i)).margin(1e-12));
            CHECK(results[2](0, i) == Approx(expected_d2N(i)).margin(1e-12));
            CHECK(results[3](0, i) == Approx(expected_d3N(i)).margin(1e-12));
        }
    }

    SECTION("Evaluation at u = 0.85") {
        Eigen::MatrixXd u(1, 1); u << 0.85;
        auto results = curve.eval_shape_functions(u, 3);

        Eigen::Vector4d expected_N(0.0033750000000000, 0.0573750000000000, 0.3251250000000000, 0.6141249999999999);
        Eigen::Vector4d expected_dN(-0.0203825545637234, -0.2106197304918083, -0.4235041892684746, 0.6545064743240061);
        Eigen::Vector4d expected_d2N(0.0894507953662083, 0.3772321703322536, -0.6945105890554222, 0.2278276233569604);
        Eigen::Vector4d expected_d3N(-0.2032875175830995, 0.6968774021132930, 1.4888884361605246, -1.9824783206907182);

        for(int i=0; i<4; ++i) {
            CHECK(results[0](0, i) == Approx(expected_N(i)).margin(1e-12));
            CHECK(results[1](0, i) == Approx(expected_dN(i)).margin(1e-12));
            CHECK(results[2](0, i) == Approx(expected_d2N(i)).margin(1e-12));
            CHECK(results[3](0, i) == Approx(expected_d3N(i)).margin(1e-12));
        }
    }

    SECTION("Partition of Unity") {
        Eigen::MatrixXd u(1, 1); u << 0.42;
        auto results = curve.eval_shape_functions(u, 3);

        CHECK(results[0].row(0).sum() == Approx(1.0).margin(1e-14));
        
        CHECK(results[1].row(0).sum() == Approx(0.0).margin(1e-14));
        CHECK(results[2].row(0).sum() == Approx(0.0).margin(1e-14));
        CHECK(results[3].row(0).sum() == Approx(0.0).margin(1e-14));
    }
}
