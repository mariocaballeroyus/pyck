#include "catch.hpp"

#include <cmath>
#include <vector>

#include "bspline.hpp"
#include "nurbs.hpp"
#include "knot_vector.hpp"

using namespace pyck;

// Count occurrences of a knot value in a basis's knot vector.
static Index multiplicity(const Basis<double>& b, double u)
{
    const KnotVector<double>& kv = b.knot_vector();
    Index m = 0;
    for (Index i = 0; i < kv.size(); ++i)
        if (kv[i] == u) ++m;
    return m;
}

/**
 * A C²-cubic B-spline with several interior knots extracts to one Bernstein
 * segment per non-empty span (n_ext = n_e·p + 1), every interior knot reaches
 * multiplicity p, and the extracted control polygon reproduces the curve to
 * machine precision.
 */
TEST_CASE("Bezier extraction: B-spline reproduces geometry", "[basis][bezier]")
{
    const Index p = 3;
    BSpline<double> bs = BSpline<double>::clamped_uniform(p, 7);  // 3 interior knots → 4 spans

    Eigen::MatrixX3d cps(7, 3);
    for (Index i = 0; i < 7; ++i)
        cps.row(i) << double(i), std::sin(double(i)), 0.3 * double(i * i);

    auto [ext, C] = bs.bezier_extract();

    const Index n_e = 4;
    REQUIRE(ext->num_basis() == n_e * p + 1);
    REQUIRE(C.rows() == ext->num_basis());
    REQUIRE(C.cols() == bs.num_basis());

    // Every interior knot now sits at full multiplicity p.
    for (double u : {0.25, 0.5, 0.75})
        REQUIRE(multiplicity(*ext, u) == p);

    Eigen::VectorXd samples(6);
    samples << 0.05, 0.2, 0.4, 0.55, 0.78, 0.95;

    Eigen::MatrixX3d cps_ext = C * cps;
    Eigen::MatrixXd R_old = bs.eval_all(samples, 0)[0];
    Eigen::MatrixXd R_new = ext->eval_all(samples, 0)[0];
    Eigen::MatrixX3d c_old = R_old * cps;
    Eigen::MatrixX3d c_new = R_new * cps_ext;
    REQUIRE((c_old - c_new).cwiseAbs().maxCoeff() < 1e-12);
}

/**
 * On a rational basis the extraction is rational-aware: the returned basis is a
 * NURBS carrying the extracted per-segment weights, and the rational transform
 * reproduces the exact circular arc.
 */
TEST_CASE("Bezier extraction: NURBS arc stays rational and exact", "[basis][bezier]")
{
    const Index p = 2;
    KnotVector<double> kv(std::vector<double>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
    Vector<double> w(3);
    w << 1.0, std::sqrt(2.0) / 2.0, 1.0;
    NURBS<double> nb(p, kv, w);

    Eigen::MatrixX3d cps(3, 3);
    cps << 1.0, 0.0, 0.0,
           1.0, 1.0, 0.0,
           0.0, 1.0, 0.0;

    // Split the single-span arc so extraction has real work to do.
    auto [split, T_split] = nb.insert_knot(0.5);
    Eigen::MatrixX3d cps_split = T_split * cps;

    auto [ext, C] = split->bezier_extract();

    REQUIRE(multiplicity(*ext, 0.5) == p);     // C⁰ between the two Bezier segments
    REQUIRE(ext->num_basis() == 2 * p + 1);    // two segments

    // The extracted basis must still be a NURBS with one weight per CP.
    auto* nb_ext = dynamic_cast<NURBS<double>*>(ext.get());
    REQUIRE(nb_ext != nullptr);
    REQUIRE(nb_ext->weights().size() == ext->num_basis());

    Eigen::VectorXd samples(6);
    samples << 0.05, 0.2, 0.4, 0.55, 0.78, 0.95;

    Eigen::MatrixX3d cps_ext = C * cps_split;
    Eigen::MatrixX3d c_arc = nb.eval_all(samples, 0)[0] * cps;
    Eigen::MatrixX3d c_ext = ext->eval_all(samples, 0)[0] * cps_ext;
    REQUIRE((c_arc - c_ext).cwiseAbs().maxCoeff() < 1e-12);

    // Every extracted sample still lies on the unit circle.
    for (Index i = 0; i < c_ext.rows(); ++i)
        REQUIRE(c_ext.row(i).head<2>().norm() == Approx(1.0).margin(1e-12));
}

/**
 * A basis with no interior knots is already a single Bezier segment: extraction
 * is the identity and preserves the concrete (rational) type and its weights.
 */
TEST_CASE("Bezier extraction: single-span basis is identity", "[basis][bezier]")
{
    const Index p = 2;
    KnotVector<double> kv(std::vector<double>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
    Vector<double> w(3);
    w << 1.0, std::sqrt(2.0) / 2.0, 1.0;
    NURBS<double> nb(p, kv, w);

    auto [ext, C] = nb.bezier_extract();

    REQUIRE(ext->num_basis() == nb.num_basis());
    REQUIRE(C.rows() == nb.num_basis());
    REQUIRE(C.cols() == nb.num_basis());
    REQUIRE((C - Eigen::MatrixXd::Identity(3, 3)).cwiseAbs().maxCoeff() < 1e-15);

    auto* nb_ext = dynamic_cast<NURBS<double>*>(ext.get());
    REQUIRE(nb_ext != nullptr);
    REQUIRE((nb_ext->weights() - w).cwiseAbs().maxCoeff() < 1e-15);
}
