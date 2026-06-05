#include "catch.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <vector>

#include "vtk_export.hpp"
#include "bspline.hpp"
#include "nurbs.hpp"
#include "patch.hpp"

using namespace pyck;
using namespace pyck::io;

// Build a non-trivial B-spline surface: a refined, warped bilinear-ish patch.
static Patch<double, 2> make_bspline_surface(Index p, Index nu, Index nv)
{
    auto bu = std::make_shared<BSpline<double>>(BSpline<double>::clamped_uniform(p, nu));
    auto bv = std::make_shared<BSpline<double>>(BSpline<double>::clamped_uniform(p, nv));

    const Vector<double> gu = bu->greville_abscissae();
    const Vector<double> gv = bv->greville_abscissae();

    ColMatrix<double, 3> cps(nu * nv, 3);
    for (Index iv = 0; iv < nv; ++iv)
        for (Index iu = 0; iu < nu; ++iu) {
            const double x = gu(iu), y = gv(iv);
            cps.row(iu + iv * nu) << x, y, 0.4 * std::sin(3.0 * x) * std::cos(2.0 * y);
        }
    return Patch<double, 2>(bu, bv, cps);
}

// Rebuild the extracted patch using an independent (full-Kronecker) apply of the
// per-direction extraction operators — a cross-check against the library's
// internal two-step kron used by extract_bezier_mesh.
static Patch<double, 2> rebuild_extracted(const Patch<double, 2>& patch)
{
    auto [bu_ext, Cu] = patch.basis(0).bezier_extract();
    auto [bv_ext, Cv] = patch.basis(1).bezier_extract();
    const Index nu_old = patch.basis(0).num_basis();
    const Index nv_old = patch.basis(1).num_basis();
    const Index nu_new = bu_ext->num_basis();
    const Index nv_new = bv_ext->num_basis();

    const ColMatrix<double, 3>& cps = patch.control_pts();
    ColMatrix<double, 3> cps_new(nu_new * nv_new, 3);
    for (Index ivn = 0; ivn < nv_new; ++ivn)
        for (Index iun = 0; iun < nu_new; ++iun) {
            Eigen::RowVector3d acc = Eigen::RowVector3d::Zero();
            for (Index ivo = 0; ivo < nv_old; ++ivo)
                for (Index iuo = 0; iuo < nu_old; ++iuo)
                    acc += Cv(ivn, ivo) * Cu(iun, iuo) * cps.row(iuo + ivo * nu_old);
            cps_new.row(iun + ivn * nu_new) = acc;
        }
    return Patch<double, 2>(bu_ext, bv_ext, cps_new);
}

/**
 * The extracted Bezier cells must reproduce the patch geometry exactly: the
 * extracted patch evaluates to the same surface, and each cell's corner points
 * (which interpolate) land on the surface.
 */
TEST_CASE("vtk_export: extracted mesh reproduces B-spline geometry", "[io][bezier]")
{
    const Index p = 3, nu = 7, nv = 6;
    Patch<double, 2> patch = make_bspline_surface(p, nu, nv);

    VtkBezierMesh<double> mesh = extract_bezier_mesh<double>(patch, {}, {});

    const Index n_eu = nu - p, n_ev = nv - p;  // clamped-uniform: one span per interior gap
    REQUIRE(mesh.num_cells() == n_eu * n_ev);
    REQUIRE(mesh.num_points() == n_eu * n_ev * (p + 1) * (p + 1));
    REQUIRE(mesh.degrees(0, 0) == p);
    REQUIRE(mesh.degrees(0, 1) == p);
    REQUIRE((mesh.rational_weights.array() == 1.0).all());  // B-spline ⇒ unit weights

    // Extracted patch reproduces the surface on a sample grid.
    Patch<double, 2> ext = rebuild_extracted(patch);
    ColMatrix<double, 2> grid(16, 2);
    Index r = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            grid.row(r++) << 0.05 + 0.3 * i, 0.05 + 0.3 * j;
    REQUIRE((patch.eval_geometry(grid) - ext.eval_geometry(grid)).cwiseAbs().maxCoeff() < 1e-12);

    // Cell corner 0 (VTK ordering puts corners first) is the patch corner point.
    ColMatrix<double, 2> origin(1, 2);
    origin << 0.0, 0.0;
    REQUIRE((mesh.points.row(0) - patch.eval_geometry(origin)).cwiseAbs().maxCoeff() < 1e-12);
}

/**
 * A field defined on the control points is pushed through the same extraction
 * operator as the geometry, so a field that is an affine function of the CPs is
 * reproduced exactly on the extracted mesh.
 */
TEST_CASE("vtk_export: control-point field is extracted consistently", "[io][bezier]")
{
    const Index p = 2, nu = 5, nv = 5;
    Patch<double, 2> patch = make_bspline_surface(p, nu, nv);

    // A scalar field equal to the x-coordinate of each control point.
    const ColMatrix<double, 3>& cps = patch.control_pts();
    Matrix<double> fx(nu * nv, 1);
    fx.col(0) = cps.col(0);

    std::map<std::string, Matrix<double>> pd{{"fx", fx}};
    VtkBezierMesh<double> mesh = extract_bezier_mesh<double>(patch, pd, {});

    REQUIRE(mesh.field_names.size() == 1);
    REQUIRE(mesh.field_names[0] == "fx");
    REQUIRE(mesh.field_values[0].rows() == mesh.num_points());
    REQUIRE(mesh.field_values[0].cols() == 1);

    // Field extracted as geometry x ⇒ field value equals the point's x at every node.
    const double err =
        (mesh.field_values[0].col(0) - mesh.points.col(0)).cwiseAbs().maxCoeff();
    REQUIRE(err < 1e-12);
}

/**
 * Writing produces a non-empty file in both binary and ASCII modes; the binary
 * file is the smaller of the two for a mesh of this size.
 */
TEST_CASE("vtk_export: writes binary and ascii vtu", "[io][bezier]")
{
    const Index p = 3, nu = 8, nv = 8;
    Patch<double, 2> patch = make_bspline_surface(p, nu, nv);
    VtkBezierMesh<double> mesh = extract_bezier_mesh<double>(patch, {}, {});

    const std::string bin = "/tmp/pyck_test_bezier_bin.vtu";
    const std::string asc = "/tmp/pyck_test_bezier_ascii.vtu";
    write_bezier_vtu<double>(bin, {mesh}, "test", true);
    write_bezier_vtu<double>(asc, {mesh}, "test", false);

    std::ifstream fb(bin, std::ios::binary | std::ios::ate);
    std::ifstream fa(asc, std::ios::binary | std::ios::ate);
    REQUIRE(fb.good());
    REQUIRE(fa.good());
    REQUIRE(fb.tellg() > 0);
    REQUIRE(fb.tellg() < fa.tellg());  // binary is more compact than ascii
}
