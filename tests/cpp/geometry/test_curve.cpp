#include "catch.hpp"

#include <cmath>
#include <string>

#include "patch.hpp"
#include "basis_derivs.hpp"
#include "intrinsic_geometry.hpp"
#include "bspline.hpp"
#include "knots.hpp"

using namespace pyck;

namespace {

// Helper: from raw parametric basis derivatives and the per-qp local frame /
// Christoffel symbol, compose the metric-normalised covariant derivatives
//   N_{,x}   = N_{,u}   / √g_{11}
//   N_{;xx}  = (N_{,uu}  - Γ N_{,u})                     / g_{11}
//   N_{;xxx} = (N_{,uuu} - 3 Γ N_{,uu} + (2 Γ² - Γ_u) N_{,u}) / g_{11}^{3/2}
// where Γ_u is derived in closed form from the local frame:
//   Γ_u = (a_{11}·a_{11} + a_1·a_{111}) / g_{11} - 2 Γ².
struct CurveDerivs
{
    Eigen::RowVectorXd N, dN, d2N, d3N;
};

CurveDerivs metric_derivs(const BasisDerivs<double, 1>& basis,
                          IntrinsicGeometry<double, 1>& local,
                          Index q)
{
    const double g11      = local.g[0][0](q);
    const double sqrt_g11 = std::sqrt(g11);
    const double g11_15   = g11 * sqrt_g11;
    local.compute_christoffels();
    const auto& chr = local.chr;
    const double Gamma    = chr.Gamma[0][0][0](q);

    const double a11_dot_a11 = local.a_d1[0][0].row(q).squaredNorm();
    const double a1_dot_a111 = local.a[0].row(q).dot(local.a_d2[0][0][0].row(q));
    const double Gamma_u     = (a11_dot_a11 + a1_dot_a111) / g11 - 2.0 * Gamma * Gamma;

    CurveDerivs d;
    d.N   = basis.N.row(q);
    d.dN  = basis.N_d1[0].row(q) / sqrt_g11;
    d.d2N = (basis.N_d2[0][0].row(q) - Gamma * basis.N_d1[0].row(q)) / g11;
    d.d3N = (basis.N_d3[0][0][0].row(q)
             - 3.0 * Gamma * basis.N_d2[0][0].row(q)
             + (2.0 * Gamma * Gamma - Gamma_u) * basis.N_d1[0].row(q)) / g11_15;
    return d;
}

} // namespace


TEST_CASE("Patch<double, 1>: Analytical Push-Forward Verification", "[geometry][curve]") {
    Vector<double> knots_vec(8);
    knots_vec << 0, 0, 0, 0, 1, 1, 1, 1;
    auto basis_ptr = std::make_shared<BSpline<double>>(3, KnotVector<double>(knots_vec));

    Eigen::MatrixXd control_pts(4, 3);
    control_pts.row(0) << 0.0,  0.0, 0.0;
    control_pts.row(1) << 1.0,  2.0, 0.0;
    control_pts.row(2) << 2.0, -1.0, 1.0;
    control_pts.row(3) << 3.0,  0.0, 0.0;

    Patch<double, 1> curve(basis_ptr, control_pts);
    const Index span = 3;  // only non-degenerate span for p=3, n=4, clamped knots

    std::vector<double> test_pts = {0.15, 0.5, 0.85};

    for (double u : test_pts)
    {
        SECTION("Evaluation at u = " + std::to_string(u)) {
            ColMatrix<double, 1> params(1, 1);
            params << u;

            // Analytical raw basis derivatives (Bernstein cubic).
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

            // Analytical curve geometry.
            Eigen::Vector3d x_1(3.0, 27*u*u - 30*u + 6, -9*u*u + 6*u);
            Eigen::Vector3d x_11(0.0, 54*u - 30, -18*u + 6);
            Eigen::Vector3d x_111(0.0, 54.0, -18.0);

            const double g11      = x_1.dot(x_1);
            const double sqrt_g11 = std::sqrt(g11);
            const double Gamma    = x_1.dot(x_11) / g11;
            const double Gamma_u  = (x_11.dot(x_11) + x_1.dot(x_111)) / g11
                                  - 2.0 * Gamma * Gamma;

            // New-API evaluation.
            auto b     = eval_basis(curve, params, span, 3);
            auto act   = curve.active_control_pts(span);
            IntrinsicGeometry local(b, act);            auto md    = metric_derivs(b, local, 0);

            CHECK(local.jac(0) == Approx(sqrt_g11).margin(1e-12));

            for (int a = 0; a < 4; ++a) {
                const double expected_dN  = N_u[a] / sqrt_g11;
                const double expected_d2N = (N_uu[a] - Gamma * N_u[a]) / g11;
                const double expected_d3N = (N_uuu[a]
                                             - 3.0 * Gamma * N_uu[a]
                                             + (2.0 * Gamma * Gamma - Gamma_u) * N_u[a])
                                             / std::pow(g11, 1.5);
                CHECK(md.dN(a)  == Approx(expected_dN).margin(1e-10));
                CHECK(md.d2N(a) == Approx(expected_d2N).margin(1e-10));
                CHECK(md.d3N(a) == Approx(expected_d3N).margin(1e-10));
            }

            // Partition of unity: derivatives of constant 1 vanish.
            CHECK(md.dN.sum()  == Approx(0.0).margin(1e-11));
            CHECK(md.d2N.sum() == Approx(0.0).margin(1e-11));
            CHECK(md.d3N.sum() == Approx(0.0).margin(1e-11));
        }
    }
}


TEST_CASE("Patch<double, 1>: External AD Numerical Validation", "[geometry][curve]") {
    Vector<double> knots_vec(8);
    knots_vec << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
    auto basis_ptr = std::make_shared<BSpline<double>>(3, KnotVector<double>(knots_vec));

    Eigen::MatrixXd control_pts(4, 3);
    control_pts.row(0) << 0.0,  0.0, 0.0;
    control_pts.row(1) << 1.0,  2.0, 0.0;
    control_pts.row(2) << 2.0, -1.0, 1.0;
    control_pts.row(3) << 3.0,  0.0, 0.0;

    Patch<double, 1> curve(basis_ptr, control_pts);
    const Index span = 3;
    auto act = curve.active_control_pts(span);

    auto eval_at = [&](double u) {
        ColMatrix<double, 1> params(1, 1); params << u;
        auto b     = eval_basis(curve, params, span, 3);
        IntrinsicGeometry local(b, act);        return metric_derivs(b, local, 0);
    };

    SECTION("Evaluation at u = 0.15") {
        auto md = eval_at(0.15);
        Eigen::Vector4d expected_N(0.6141250000000000, 0.3251250000000000, 0.0573750000000000, 0.0033750000000000);
        Eigen::Vector4d expected_dN(-0.5807828087105747, 0.3758006409303719, 0.1868955059172438, 0.0180866618629591);
        Eigen::Vector4d expected_d2N(-0.1238056625562066, -0.3506754246294261, 0.3946046702878425, 0.0798764168977904);
        Eigen::Vector4d expected_d3N(0.9294862062463560, -1.4201195886353655, 0.2153416673286355, 0.2752917150603740);
        for (int i = 0; i < 4; ++i) {
            CHECK(md.N  (i) == Approx(expected_N  (i)).margin(1e-12));
            CHECK(md.dN (i) == Approx(expected_dN (i)).margin(1e-12));
            CHECK(md.d2N(i) == Approx(expected_d2N(i)).margin(1e-12));
            CHECK(md.d3N(i) == Approx(expected_d3N(i)).margin(1e-12));
        }
    }

    SECTION("Evaluation at u = 0.5") {
        auto md = eval_at(0.5);
        Eigen::Vector4d expected_N(0.1250000000000000, 0.3750000000000000, 0.3750000000000000, 0.1250000000000000);
        Eigen::Vector4d expected_dN(-0.1961161351381840, -0.1961161351381840, 0.1961161351381840, 0.1961161351381840);
        Eigen::Vector4d expected_d2N(0.2209072978303747, -0.1893491124260355, -0.2209072978303747, 0.1893491124260355);
        Eigen::Vector4d expected_d3N(-0.2691451698330937, 0.2589887483299580, -0.1599636386743859, 0.1701200601775215);
        for (int i = 0; i < 4; ++i) {
            CHECK(md.N  (i) == Approx(expected_N  (i)).margin(1e-12));
            CHECK(md.dN (i) == Approx(expected_dN (i)).margin(1e-12));
            CHECK(md.d2N(i) == Approx(expected_d2N(i)).margin(1e-12));
            CHECK(md.d3N(i) == Approx(expected_d3N(i)).margin(1e-12));
        }
    }

    SECTION("Evaluation at u = 0.85") {
        auto md = eval_at(0.85);
        Eigen::Vector4d expected_N(0.0033750000000000, 0.0573750000000000, 0.3251250000000000, 0.6141249999999999);
        Eigen::Vector4d expected_dN(-0.0203825545637234, -0.2106197304918083, -0.4235041892684746, 0.6545064743240061);
        Eigen::Vector4d expected_d2N(0.0894507953662083, 0.3772321703322536, -0.6945105890554222, 0.2278276233569604);
        Eigen::Vector4d expected_d3N(-0.2032875175830995, 0.6968774021132930, 1.4888884361605246, -1.9824783206907182);
        for (int i = 0; i < 4; ++i) {
            CHECK(md.N  (i) == Approx(expected_N  (i)).margin(1e-12));
            CHECK(md.dN (i) == Approx(expected_dN (i)).margin(1e-12));
            CHECK(md.d2N(i) == Approx(expected_d2N(i)).margin(1e-12));
            CHECK(md.d3N(i) == Approx(expected_d3N(i)).margin(1e-12));
        }
    }

    SECTION("Partition of Unity") {
        auto md = eval_at(0.42);
        CHECK(md.N  .sum() == Approx(1.0).margin(1e-14));
        CHECK(md.dN .sum() == Approx(0.0).margin(1e-14));
        CHECK(md.d2N.sum() == Approx(0.0).margin(1e-14));
        CHECK(md.d3N.sum() == Approx(0.0).margin(1e-14));
    }
}
