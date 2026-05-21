#include "catch.hpp"

#include <cmath>
#include <memory>
#include <vector>
#include <Eigen/Eigenvalues>

#include "patch.hpp"
#include "factories.hpp"
#include "bspline.hpp"
#include "knot_vector.hpp"
#include "gauss_legendre.hpp"
#include "plane_stress_2d.hpp"
#include "plane_stress_2d.hpp"
#include "plate_reissner_mindlin_3p.hpp"
#include "shell_reissner_mindlin_5p.hpp"

using namespace pyck;

namespace {

// Sum the local stiffness over all non-zero spans of a patch using p+1 Gauss.
template <std::floating_point T>
Matrix<T> assemble_global_K(const Patch<T, 2>& patch,
                            const Element<T, 2>& element,
                            int gauss_p)
{
    const Index ndof = element.num_node_dofs();
    const Index ncps = patch.num_control_pts();
    Matrix<T> K = Matrix<T>::Zero(ndof * ncps, ndof * ncps);

    GaussLegendre<T, 2> quad(gauss_p);
    PatchValues<T, 2> pv(patch,
                         static_cast<Index>(element.min_order()), quad);
    const std::size_t num_live = static_cast<std::size_t>(pv.num_elements());

    Matrix<T> Ke;
    for (std::size_t e = 0; e < num_live; ++e) {
        pv.reinit(e);
        element.compute_local_stiffness(pv, Ke);

        const auto& dofs = pv.elem_nodes;
        for (Index i = 0; i < (Index)dofs.size(); ++i) {
            for (Index j = 0; j < (Index)dofs.size(); ++j) {
                for (Index a = 0; a < ndof; ++a) {
                    for (Index b = 0; b < ndof; ++b) {
                        K(ndof * dofs[i] + a, ndof * dofs[j] + b)
                            += Ke(ndof * i + a, ndof * j + b);
                    }
                }
            }
        }
    }
    return K;
}

} // namespace


// ===========================================================================
// Test 1 — Symmetry & rigid-body modes on a flat unit square
// ===========================================================================

TEST_CASE("ShellRM5p: K is symmetric and has 6 rigid-body modes (flat patch)",
          "[elements][shell-rm5p]") {

    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis = std::make_shared<BSpline<double>>(2, kv);
    auto surf  = rectangle<double>(basis, basis, 1.0, 1.0);
    auto patch = std::make_shared<Patch<double, 2>>(surf);

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.05);
    ShellReissnerMindlin5p<double> element(material);

    Matrix<double> K = assemble_global_K(*patch, element, 3);

    // Symmetry
    REQUIRE(K.allFinite());
    CHECK((K - K.transpose()).norm() / K.norm() < 1e-12);

    // Rigid-body modes: 3 translations × constant {u_x=1, u_y=1, u_z=1} fields,
    // 3 rotations × infinitesimal-rotation fields. We check Kv = 0 for the
    // 3 translational modes (the rotation modes also need θ updates and are
    // checked separately in the eigenvalue test).
    const Index ncp = patch->num_control_pts();
    const Index N = 5 * ncp;
    for (int axis = 0; axis < 3; ++axis) {
        Vector<double> v = Vector<double>::Zero(N);
        for (Index i = 0; i < ncp; ++i) v(5 * i + axis) = 1.0;
        Vector<double> Kv = K * v;
        CHECK(Kv.norm() < 1e-8 * K.norm());
    }
}


// ===========================================================================
// Test 2 — Eigenvalue spectrum: exactly 6 zero eigenvalues
// ===========================================================================

TEST_CASE("ShellRM5p: K has exactly 6 near-zero eigenvalues (flat patch)",
          "[elements][shell-rm5p]") {

    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis = std::make_shared<BSpline<double>>(2, kv);
    auto surf  = rectangle<double>(basis, basis, 1.0, 1.0);
    auto patch = std::make_shared<Patch<double, 2>>(surf);

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.05);
    ShellReissnerMindlin5p<double> element(material);

    Matrix<double> K = assemble_global_K(*patch, element, 3);

    Eigen::SelfAdjointEigenSolver<Matrix<double>> es(K);
    REQUIRE(es.info() == Eigen::Success);
    Vector<double> eig = es.eigenvalues();

    // The free 5p shell has 6 rigid-body modes (3 trans + 3 rot) on a flat
    // patch — the drilling DOF about a3 is genuinely missing but is not a
    // rigid-body mode.
    const double tol = 1e-6 * eig.maxCoeff();
    Index num_zero = 0;
    for (Index i = 0; i < eig.size(); ++i) if (std::abs(eig(i)) < tol) ++num_zero;
    CHECK(num_zero == 6);
    // Remaining eigenvalues are strictly positive.
    for (Index i = num_zero; i < eig.size(); ++i) {
        CHECK(eig(i) > 0.0);
    }
}


// ===========================================================================
// Test 3 — Flat-plate reduction: bending+shear block matches PlateRM3p
// ===========================================================================

TEST_CASE("ShellRM5p reduces to PlateRM3p on a flat axis-aligned patch",
          "[elements][shell-rm5p]") {

    // p=2, n=4 quadratic patch on the unit square.
    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis = std::make_shared<BSpline<double>>(2, kv);
    auto surf  = rectangle<double>(basis, basis, 1.0, 1.0);
    auto patch = std::make_shared<Patch<double, 2>>(surf);
    const Index ncp = patch->num_control_pts();

    const double E = 1.0e6, nu = 0.3, t = 0.05, k_s = 5.0 / 6.0;

    // Plate-side material/element (flat-plate API)
    auto plate_mat = std::make_shared<PlaneStress2d<double>>(E, nu, t, k_s);
    PlateReissnerMindlin3p<double> plate_el(plate_mat);

    // Shell-side material/element
    auto shell_mat = std::make_shared<PlaneStress2d<double>>(E, nu, t, k_s);
    ShellReissnerMindlin5p<double> shell_el(shell_mat);

    Matrix<double> Kp = assemble_global_K(*patch, plate_el,  3);  // (3·N) × (3·N)
    Matrix<double> Ks = assemble_global_K(*patch, shell_el,  3);  // (5·N) × (5·N)

    // Plate slot order : (w, θ_x, θ_y) per node.
    // Shell slot order: (u_x, u_y, u_z, θ_1, θ_2) per node.
    // On a flat z=0 axis-aligned patch:
    //   w  ↔ u_z  (slot 2)
    //   θ_x = ∂w/∂x rotation around the y-axis  ↔  θ_2  (slot 4)  — see below
    //   θ_y = ∂w/∂y rotation around the x-axis  ↔  −θ_1 (slot 3)
    // …or some sign/index permutation. Rather than chase conventions,
    // verify the *active 3×3 sub-block on the (u_z, θ_1, θ_2) DOFs*
    // is symmetric-positive-definite of the expected dimension and that
    // the in-plane (u_x, u_y) and bending-coupling sub-blocks are
    // independent (zero coupling for a flat plate).
    //
    // Concretely we check that on a flat plate the shell K decouples:
    //   - The (u_x, u_y) × (u_z, θ_1, θ_2) cross-block is zero.
    //   - The (u_x, u_y) sub-block is strictly positive (membrane stiffness).
    //   - The (u_z, θ_1, θ_2) bending-shear sub-block is strictly positive.
    Matrix<double> K_membrane = Matrix<double>::Zero(2 * ncp, 2 * ncp);
    Matrix<double> K_bending  = Matrix<double>::Zero(3 * ncp, 3 * ncp);
    Matrix<double> K_cross    = Matrix<double>::Zero(2 * ncp, 3 * ncp);

    for (Index i = 0; i < ncp; ++i) {
        for (Index j = 0; j < ncp; ++j) {
            // Membrane block: (u_x, u_y) of node i × (u_x, u_y) of node j.
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    K_membrane(2*i+a, 2*j+b) = Ks(5*i+a, 5*j+b);
            // Bending+shear block: (u_z, θ_1, θ_2) × (u_z, θ_1, θ_2).
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    K_bending(3*i+a, 3*j+b) = Ks(5*i + 2 + a, 5*j + 2 + b);
            // Cross block: (u_x, u_y) × (u_z, θ_1, θ_2).
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 3; ++b)
                    K_cross(2*i+a, 3*j+b) = Ks(5*i+a, 5*j + 2 + b);
        }
    }

    // 1) Cross-block is zero: in-plane and bending modes decouple on a flat plate.
    CHECK(K_cross.norm() / Ks.norm() < 1e-12);

    // 2) Membrane sub-block is symmetric and has 3 zero eigenvalues
    //    (2 in-plane translations + 1 in-plane rotation). The remaining are positive.
    CHECK((K_membrane - K_membrane.transpose()).norm() / (K_membrane.norm() + 1e-30) < 1e-12);
    Eigen::SelfAdjointEigenSolver<Matrix<double>> esm(K_membrane);
    REQUIRE(esm.info() == Eigen::Success);
    Vector<double> em = esm.eigenvalues();
    const double tolm = 1e-6 * em.maxCoeff();
    int num_zero_m = 0;
    for (Index i = 0; i < em.size(); ++i) if (std::abs(em(i)) < tolm) ++num_zero_m;
    CHECK(num_zero_m == 3);

    // 3) Bending+shear sub-block is symmetric and has 3 zero eigenvalues
    //    (transverse translation + 2 rotations about in-plane axes).
    CHECK((K_bending - K_bending.transpose()).norm() / (K_bending.norm() + 1e-30) < 1e-12);
    Eigen::SelfAdjointEigenSolver<Matrix<double>> esb(K_bending);
    REQUIRE(esb.info() == Eigen::Success);
    Vector<double> eb = esb.eigenvalues();
    const double tolb = 1e-6 * eb.maxCoeff();
    int num_zero_b = 0;
    for (Index i = 0; i < eb.size(); ++i) if (std::abs(eb(i)) < tolb) ++num_zero_b;
    CHECK(num_zero_b == 3);

    // 4) The (u_z, θ_1, θ_2) bending+shear block of the shell should match
    //    the plate K with a slot-permutation on the rotations. We check the
    //    spectra agree (eigenvalues are permutation-invariant).
    Eigen::SelfAdjointEigenSolver<Matrix<double>> esp(Kp);
    REQUIRE(esp.info() == Eigen::Success);
    Vector<double> ep = esp.eigenvalues();

    REQUIRE(eb.size() == ep.size());
    for (Index i = 0; i < ep.size(); ++i) {
        CHECK(eb(i) == Approx(ep(i)).margin(1e-8 * ep.cwiseAbs().maxCoeff()));
    }
}


// ===========================================================================
// Test 4 — Curved-shell sanity: K symmetric, PSD, 3 exact translation modes.
//
// Note: on a curved shell the 3 rotational rigid-body modes have non-polynomial
// θ_α amplitudes (∝ 1/√(1+|∇x|²)), so a bilinear basis cannot represent them
// exactly. The discretization therefore exhibits 3 exact zero modes
// (translations) and 3 small-but-nonzero "approximate" rotational modes that
// converge to zero with mesh/order refinement. We assert the matrix is PSD
// and that translational modes annihilate K exactly.
// ===========================================================================

TEST_CASE("ShellRM5p: curved twisted patch K is SPSD; translations are exact RBMs",
          "[elements][shell-rm5p]") {

    auto kv = KnotVector<double>(std::vector<double>{0, 0, 1, 1});
    auto basis = std::make_shared<BSpline<double>>(1, kv);

    Eigen::MatrixXd cp(4, 3);
    cp.row(0) << 0.0, 0.0, 0.0;
    cp.row(1) << 1.0, 0.0, 0.0;
    cp.row(2) << 0.0, 1.0, 0.0;
    cp.row(3) << 1.0, 1.0, 1.0;

    auto patch = std::make_shared<Patch<double, 2>>(basis, basis, cp);

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.05);
    ShellReissnerMindlin5p<double> element(material);

    Matrix<double> K = assemble_global_K(*patch, element, 3);

    REQUIRE(K.allFinite());
    CHECK((K - K.transpose()).norm() / K.norm() < 1e-12);

    // Translations annihilate K exactly.
    const Index ncp = patch->num_control_pts();
    const Index N = 5 * ncp;
    for (int axis = 0; axis < 3; ++axis) {
        Vector<double> v = Vector<double>::Zero(N);
        for (Index i = 0; i < ncp; ++i) v(5 * i + axis) = 1.0;
        Vector<double> Kv = K * v;
        CHECK(Kv.norm() < 1e-8 * K.norm());
    }

    // Spectrum: PSD (no negative eigenvalues beyond numerical noise).
    Eigen::SelfAdjointEigenSolver<Matrix<double>> es(K);
    REQUIRE(es.info() == Eigen::Success);
    Vector<double> eig = es.eigenvalues();
    const double tol = 1e-6 * eig.maxCoeff();
    for (Index i = 0; i < eig.size(); ++i) {
        CHECK(eig(i) > -tol);
    }
    // Exactly 3 zero modes (the translations).
    Index num_zero = 0;
    for (Index i = 0; i < eig.size(); ++i) if (std::abs(eig(i)) < tol) ++num_zero;
    CHECK(num_zero == 3);
}
