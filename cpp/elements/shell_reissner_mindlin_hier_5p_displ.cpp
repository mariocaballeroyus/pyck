#include "shell_reissner_mindlin_hier_5p_displ.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlinHier5pDispl<T>::ShellReissnerMindlinHier5pDispl(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlinHier5pDispl: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellReissnerMindlinHier5pDispl<T>::membrane_strain_matrix(const ElementValues<T, 2>& ev,
                                                      Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const auto cov = ev.cov_basis(q);
        const Vector3<T> A1 = cov(0), A2 = cov(1);
        const auto Bc = ev.curvature(q);
        const T B11 = Bc(0, 0), B22 = Bc(1, 1), B12 = Bc(0, 1);
        const auto G  = ev.dN(q);
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T G1i = G(i, 0), G2i = G(i, 1), Ni = Nf(i);
            // Membrane ε_{αβ} = ½(A_α·v_{b,β}+A_β·v_{b,α}) - B_{αβ}(v^{s1}+v^{s2}). The shear
            // deflections project along A₃, so v_s,α·A_β = -B_{αβ} v_s (second fundamental form).
            for (Index k = 0; k < 3; ++k) {
                B(8*q,     5*i + k) = G1i * A1(k);
                B(8*q + 1, 5*i + k) = G2i * A2(k);
                B(8*q + 2, 5*i + k) = G1i * A2(k) + G2i * A1(k);
            }
            B(8*q,     5*i + 3) = -B11 * Ni;          B(8*q,     5*i + 4) = -B11 * Ni;
            B(8*q + 1, 5*i + 3) = -B22 * Ni;          B(8*q + 1, 5*i + 4) = -B22 * Ni;
            B(8*q + 2, 5*i + 3) = -T(2) * B12 * Ni;   B(8*q + 2, 5*i + 4) = -T(2) * B12 * Ni;
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pDispl<T>::bending_strain_matrix(const ElementValues<T, 2>& ev,
                                                     Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        // Third fundamental form (B²)_{αβ} = A_{3,α}·A_{3,β}.
        const auto nd = ev.normal_deriv(q);
        const Vector3<T> A3_d1 = nd(0), A3_d2 = nd(1);
        const T B2_11 = A3_d1.dot(A3_d1), B2_22 = A3_d2.dot(A3_d2), B2_12 = A3_d1.dot(A3_d2);

        const operators::CovariantHessian<T, 2> hess{ev, q};
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T H11 = hess(i, 0, 0), H22 = hess(i, 1, 1), H12 = hess(i, 0, 1);
            const T Ni = Nf(i);

            // (a) Kirchhoff-Love bending of the bending displacement v_b: κ = -v_{b|αβ}·A₃
            //     (Cartesian slots 0..2, covariant Hessian).
            for (Index k = 0; k < 3; ++k) {
                B(8*q + 3, 5*i + k) =        -H11 * A3(k);
                B(8*q + 4, 5*i + k) =        -H22 * A3(k);
                B(8*q + 5, 5*i + k) = -T(2) * H12 * A3(k);
            }

            // (b) CROSSED shear bending (covariant Hessian), from Oesterle Eqs. (18)/(27):
            //     φ̃¹ carries (v_b+v_s1),₂ and φ̃² carries (v_b+v_s2),₁, so κ₁₁ bends with
            //     v^{s2} and κ₂₂ with v^{s1}. The crossing is what removes the in-plane
            //     zero-energy shear modes (v^{s1}=f(ξ²) etc.).
            B(8*q + 3, 5*i + 4) = -H11;   // v^{s2}: κ₁₁
            B(8*q + 5, 5*i + 4) = -H12;   //         2κ₁₂
            B(8*q + 4, 5*i + 3) = -H22;   // v^{s1}: κ₂₂
            B(8*q + 5, 5*i + 3) = -H12;   //         2κ₁₂

            // (c) Shear-deflection curvature coupling (curved shell only): the v_s displace
            //     along A₃, so the displacement-curvature term v_s,α·A_{3,β} = (B²)_{αβ} v_s
            //     (third fundamental form). Vanishes on a flat patch (A_{3,α}=0).
            B(8*q + 3, 5*i + 3) +=        Ni * B2_11;   B(8*q + 3, 5*i + 4) +=        Ni * B2_11;
            B(8*q + 4, 5*i + 3) +=        Ni * B2_22;   B(8*q + 4, 5*i + 4) +=        Ni * B2_22;
            B(8*q + 5, 5*i + 3) += T(2) * Ni * B2_12;   B(8*q + 5, 5*i + 4) += T(2) * Ni * B2_12;
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pDispl<T>::shear_strain_matrix(const ElementValues<T, 2>& ev,
                                                   Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const auto G = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T G1i = G(i, 0), G2i = G(i, 1);
            // Transverse shear 2ε_{α3} = v^{sα}_{,α} (Oesterle Eqs. 32-33): the v_sα displace
            // along A₃, so (v^{sα}A₃)_{,α}·A₃ = v^{sα}_{,α} — already the covariant strain
            // component. Each component depends on ONE shear DOF only; lowering through the
            // metric (as the contravariant difference vector of Hier5p requires) would couple
            // the DOFs and lose the decoupled/solenoidal shear modes of the formulation.
            B(8*q + 6, 5*i + 3) = G1i;   // 2ε₁₃ = v^{s1}_{,1}
            B(8*q + 7, 5*i + 4) = G2i;   // 2ε₂₃ = v^{s2}_{,2}
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlinHier5pDispl<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    // Transform elasticity matrices into curvilinear coordinate system
    const StaticVector<T, 3> metric_inv = ev.metric_inv_voigt(q);
    const Eigen::Matrix<T, 3, 3> C  = material_->elasticity_voigt(metric_inv);
    const Eigen::Matrix<T, 2, 2> Cs = material_->shear_voigt(metric_inv);

    // Scale material matrices
    const T t = material_->thickness();
    const Eigen::Matrix<T, 3, 3> Dm = t * C;
    const Eigen::Matrix<T, 3, 3> Db = (t * t * t / T(12)) * C;
    const Eigen::Matrix<T, 2, 2> Ds = Cs;

    // Assemble into full constitutive matrix
    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(8, 8);
    D.template block<3, 3>(0, 0) = Dm;
    D.template block<3, 3>(3, 3) = Db;
    D.template block<2, 2>(6, 6) = Ds;
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
ShellReissnerMindlinHier5pDispl<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset shape matrix values
    Matrix<T>& N_w = this->N_w_;
    N_w.setZero(3 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto Nf = ev.N(q);
        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);
            // Mid-surface displacement u = v_b + (v^{s1}+v^{s2})A₃: the shear deflections
            // project along the director, so they add to the recovered displacement and do
            // work against the load (Oesterle Eqs. 13-15).
            N_w(3 * q + 0, 5 * i + 0) = Ni;
            N_w(3 * q + 1, 5 * i + 1) = Ni;
            N_w(3 * q + 2, 5 * i + 2) = Ni;
            for (Index k = 0; k < 3; ++k) {
                N_w(3 * q + k, 5 * i + 3) = Ni * A3(k);
                N_w(3 * q + k, 5 * i + 4) = Ni * A3(k);
            }
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pDispl<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset shape matrix values
    Matrix<T>& N_phi = this->N_phi_;
    N_phi.setZero(2 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto G = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T G1i = G(i, 0), G2i = G(i, 1);

            // Covariant director tilt θ_α = (a₃ − A₃)·A_α = −v_b,α·A₃ + (Φ̃_s × A₃)·A_α.
            // From Oesterle Eqs. (17)-(21), the shear rotation is crossed (the J from
            // (A_β×A₃)·A_α = ±J cancels the 1/J in φ̃): θ_1^s = −v^{s2}_{,1}, θ_2^s = −v^{s1}_{,2}.
            for (Index k = 0; k < 3; ++k) {
                N_phi(2*q,     5*i + k) = -G1i * A3(k);   // -v_b,1·A₃
                N_phi(2*q + 1, 5*i + k) = -G2i * A3(k);   // -v_b,2·A₃
            }
            N_phi(2*q,     5*i + 4) = -G1i;   // θ_1 += -v^{s2}_{,1}
            N_phi(2*q + 1, 5*i + 3) = -G2i;   // θ_2 += -v^{s1}_{,2}
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pDispl<T>::director_variation(const ElementValues<T, 2>& parent,
                                                      const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    // Surface normal a_3 rotates with the bending (Cartesian) displacement only; the
    // hierarchic-displacement slots carry shear, which does not tilt the surface.
    this->surface_director_variation(parent, dir, out);
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlinHier5pDispl<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlinHier5pDispl<float>;
#endif

} // namespace pyck
