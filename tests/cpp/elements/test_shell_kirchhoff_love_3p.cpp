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
#include "shell_kirchhoff_love_3p.hpp"

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
    ElementValues<T, 2> pv(patch,
                         element.basis_order(), element.flags(), quad);
    const std::size_t num_live = static_cast<std::size_t>(pv.num_elements());

    Matrix<T> Ke;
    for (std::size_t e = 0; e < num_live; ++e) {
        pv.reinit(e);
        element.compute_local_stiffness(pv, Ke);

        const auto& dofs = pv.elem_cps_;
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
// Test 1 — Symmetry & translational rigid-body modes on a flat unit square
// ===========================================================================

TEST_CASE("ShellKL3p: K is symmetric and translations are exact RBMs (flat patch)",
          "[elements][shell-kl3p]") {

    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis = std::make_shared<BSpline<double>>(2, kv);
    auto surf  = rectangle<double>(basis, basis, 1.0, 1.0);
    auto patch = std::make_shared<Patch<double, 2>>(surf);

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.05);
    ShellKirchhoffLove3p<double> element(material);

    Matrix<double> K = assemble_global_K(*patch, element, 3);

    REQUIRE(K.allFinite());
    CHECK((K - K.transpose()).norm() / K.norm() < 1e-12);

    // 3 rigid translations (constant {u_x=1, u_y=1, u_z=1} fields) annihilate K.
    const Index ncp = patch->num_control_pts();
    const Index N = 3 * ncp;
    for (int axis = 0; axis < 3; ++axis) {
        Vector<double> v = Vector<double>::Zero(N);
        for (Index i = 0; i < ncp; ++i) v(3 * i + axis) = 1.0;
        Vector<double> Kv = K * v;
        CHECK(Kv.norm() < 1e-8 * K.norm());
    }
}


// ===========================================================================
// Test 2 — Eigenvalue spectrum: exactly 6 zero eigenvalues on a flat patch
// ===========================================================================

TEST_CASE("ShellKL3p: K has exactly 6 near-zero eigenvalues (flat patch)",
          "[elements][shell-kl3p]") {

    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis = std::make_shared<BSpline<double>>(2, kv);
    auto surf  = rectangle<double>(basis, basis, 1.0, 1.0);
    auto patch = std::make_shared<Patch<double, 2>>(surf);

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.05);
    ShellKirchhoffLove3p<double> element(material);

    Matrix<double> K = assemble_global_K(*patch, element, 3);

    Eigen::SelfAdjointEigenSolver<Matrix<double>> es(K);
    REQUIRE(es.info() == Eigen::Success);
    Vector<double> eig = es.eigenvalues();

    // With 3 Cartesian displacement DOFs every rigid-body mode is a linear
    // displacement field, exactly representable by the basis: 3 translations,
    // 2 out-of-plane rotations, 1 in-plane (drilling) rotation = 6 zero modes.
    const double tol = 1e-6 * eig.maxCoeff();
    Index num_zero = 0;
    for (Index i = 0; i < eig.size(); ++i) if (std::abs(eig(i)) < tol) ++num_zero;
    CHECK(num_zero == 6);
    for (Index i = num_zero; i < eig.size(); ++i) {
        CHECK(eig(i) > 0.0);
    }
}


// ===========================================================================
// Test 3 — Flat-plate decoupling: membrane and transverse bending separate
//
// On a flat axis-aligned unit patch (A_{,αβ}=0, A_3=ẑ, b=0) the membrane strain
// of u_z and the bending strain of (u_x, u_y) both vanish, so the in-plane
// (u_x, u_y) and transverse u_z DOFs decouple; the u_z sub-block is the pure
// Kirchhoff–Love bending stiffness (κ = −∇²w on a flat patch).
// ===========================================================================

TEST_CASE("ShellKL3p: membrane and transverse bending decouple on a flat patch",
          "[elements][shell-kl3p]") {

    auto kv = KnotVector<double>::clamped_uniform(2, 4);
    auto basis = std::make_shared<BSpline<double>>(2, kv);
    auto surf  = rectangle<double>(basis, basis, 1.0, 1.0);
    auto patch = std::make_shared<Patch<double, 2>>(surf);
    const Index ncp = patch->num_control_pts();

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.05);
    ShellKirchhoffLove3p<double> shell_el(material);

    Matrix<double> Ks = assemble_global_K(*patch, shell_el, 3);  // 3N × 3N

    // Slot order: (u_x, u_y, u_z) per node.
    Matrix<double> K_membrane = Matrix<double>::Zero(2 * ncp, 2 * ncp);
    Matrix<double> K_uz       = Matrix<double>::Zero(ncp, ncp);
    Matrix<double> K_cross    = Matrix<double>::Zero(2 * ncp, ncp);

    for (Index i = 0; i < ncp; ++i) {
        for (Index j = 0; j < ncp; ++j) {
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    K_membrane(2*i+a, 2*j+b) = Ks(3*i+a, 3*j+b);
            K_uz(i, j) = Ks(3*i + 2, 3*j + 2);
            for (int a = 0; a < 2; ++a)
                K_cross(2*i+a, j) = Ks(3*i+a, 3*j + 2);
        }
    }

    // 1) In-plane and transverse DOFs decouple on a flat plate.
    CHECK(K_cross.norm() / Ks.norm() < 1e-12);

    // 2) Membrane sub-block is symmetric with 3 zero modes (2 translations +
    //    1 in-plane drilling rotation); the rest strictly positive.
    CHECK((K_membrane - K_membrane.transpose()).norm() / (K_membrane.norm() + 1e-30) < 1e-12);
    Eigen::SelfAdjointEigenSolver<Matrix<double>> esm(K_membrane);
    REQUIRE(esm.info() == Eigen::Success);
    Vector<double> em = esm.eigenvalues();
    const double tolm = 1e-6 * em.maxCoeff();
    int num_zero_m = 0;
    for (Index i = 0; i < em.size(); ++i) if (std::abs(em(i)) < tolm) ++num_zero_m;
    CHECK(num_zero_m == 3);

    // 3) The transverse u_z bending sub-block is symmetric and carries the 3
    //    out-of-plane rigid modes (transverse translation + 2 bending rotations).
    CHECK((K_uz - K_uz.transpose()).norm() / (K_uz.norm() + 1e-30) < 1e-12);
    Eigen::SelfAdjointEigenSolver<Matrix<double>> esb(K_uz);
    REQUIRE(esb.info() == Eigen::Success);
    Vector<double> eb = esb.eigenvalues();
    const double tolb = 1e-6 * eb.maxCoeff();
    int num_zero_b = 0;
    for (Index i = 0; i < eb.size(); ++i) if (std::abs(eb(i)) < tolb) ++num_zero_b;
    CHECK(num_zero_b == 3);
}


// ===========================================================================
// Test 4 — Curved-shell sanity: K symmetric, SPSD, 3 exact translation modes.
//
// On a curved shell the rotational rigid-body modes are non-polynomial, so a
// bilinear patch represents only the 3 translations exactly; the matrix stays
// SPSD with exactly 3 zero modes.
// ===========================================================================

TEST_CASE("ShellKL3p: curved twisted patch K is SPSD; translations are exact RBMs",
          "[elements][shell-kl3p]") {

    auto kv = KnotVector<double>::clamped_uniform(2, 3);
    auto basis = std::make_shared<BSpline<double>>(2, kv);

    // A gently curved (paraboloid-like) control net, degree 2 so the bending
    // block sees nonzero curvature.
    const Index n1 = static_cast<Index>(basis->num_basis());
    Eigen::MatrixXd cp(n1 * n1, 3);
    for (Index i = 0; i < n1; ++i) {
        for (Index j = 0; j < n1; ++j) {
            const double x = static_cast<double>(i) / (n1 - 1);
            const double y = static_cast<double>(j) / (n1 - 1);
            cp.row(i * n1 + j) << x, y, 0.2 * (x * x + y * y);
        }
    }

    auto patch = std::make_shared<Patch<double, 2>>(basis, basis, cp);

    auto material = std::make_shared<PlaneStress2d<double>>(1.0e6, 0.3, 0.05);
    ShellKirchhoffLove3p<double> element(material);

    Matrix<double> K = assemble_global_K(*patch, element, 3);

    REQUIRE(K.allFinite());
    CHECK((K - K.transpose()).norm() / K.norm() < 1e-12);

    const Index ncp = patch->num_control_pts();
    const Index N = 3 * ncp;
    for (int axis = 0; axis < 3; ++axis) {
        Vector<double> v = Vector<double>::Zero(N);
        for (Index i = 0; i < ncp; ++i) v(3 * i + axis) = 1.0;
        Vector<double> Kv = K * v;
        CHECK(Kv.norm() < 1e-8 * K.norm());
    }

    Eigen::SelfAdjointEigenSolver<Matrix<double>> es(K);
    REQUIRE(es.info() == Eigen::Success);
    Vector<double> eig = es.eigenvalues();
    const double tol = 1e-6 * eig.maxCoeff();
    for (Index i = 0; i < eig.size(); ++i) {
        CHECK(eig(i) > -tol);
    }
}
