#include "shell_reissner_mindlin_4p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"
#include "surface_geometry.hpp"
#include "laplace_beltrami.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlin4p<T>::ShellReissnerMindlin4p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlin4p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellReissnerMindlin4p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(8 * Q, 4 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = ev.results_[0].col(q);
        auto slab1 = ev.results_[1].col(q);

        const T G1_11 = ev.Gamma(0, 0, 0)(q);
        const T G1_12 = ev.Gamma(0, 0, 1)(q);
        const T G1_22 = ev.Gamma(0, 1, 1)(q);
        const T G2_11 = ev.Gamma(1, 0, 0)(q);
        const T G2_12 = ev.Gamma(1, 0, 1)(q);
        const T G2_22 = ev.Gamma(1, 1, 1)(q);

        // Midsurface metric + Jacobian (for the surface permutation ε_α^β).
        const T A11 = ev.g(0, 0)(q);
        const T A12 = ev.g(0, 1)(q);
        const T A22 = ev.g(1, 1)(q);
        const T invJ = T(1) / ev.jac(q);

        // Mixed surface permutation ε_α^β = g_{αμ} ϵ^{μβ} / √g, reducing to
        // ε_1^2 = +1, ε_2^1 = −1 on the orthonormal lamina.
        const T e1_1 = -A12 * invJ, e1_2 = A11 * invJ;
        const T e2_1 = -A22 * invJ, e2_2 = A12 * invJ;

        // Second fundamental form B_{αβ} = b_{αβ} and shape operator B_α^β.
        const T B11 = ev.b(0, 0)(q), B12 = ev.b(0, 1)(q), B22 = ev.b(1, 1)(q);
        const T Bmix11 = ev.b_mixed(0, 0)(q);  // B^1_1
        const T Bmix12 = ev.b_mixed(0, 1)(q);  // B^1_2
        const T Bmix21 = ev.b_mixed(1, 0)(q);  // B^2_1
        const T Bmix22 = ev.b_mixed(1, 1)(q);  // B^2_2
        // (B^2)_{αβ} = B_{αμ} B^μ_β.
        const T B2_11 = B11 * Bmix11 + B12 * Bmix21;
        const T B2_22 = B12 * Bmix12 + B22 * Bmix22;
        const T B2_12 = B11 * Bmix12 + B12 * Bmix22;

        // Covariant derivative of curvature (B_α^γ)_{|β}, needed by the bending
        // ∇B coupling of the deformed-director formulation. Source the partials
        // ∂_γ B_{αβ} = a_{αβγ}·A_3 + a_{αβ}·A_{3,γ} from the third position
        // derivatives and the unit-normal derivatives, then add the connection
        // terms. Vanishes (as a tensor) on constant-curvature surfaces — flat,
        // sphere, cylinder — so it does not perturb those benchmarks.
        const Eigen::Matrix<T, 3, 1> A3    = ev.n.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> A3_d1 = ev.n_d1(0).row(q).transpose();   // A_{3,1}
        const Eigen::Matrix<T, 3, 1> A3_d2 = ev.n_d1(1).row(q).transpose();   // A_{3,2}
        const Eigen::Matrix<T, 3, 1> A1_d1 = ev.a_d1(0, 0).row(q).transpose();   // A_{1,1}
        const Eigen::Matrix<T, 3, 1> A1_d2 = ev.a_d1(0, 1).row(q).transpose();   // A_{1,2} = A_{2,1}
        const Eigen::Matrix<T, 3, 1> A2_d2 = ev.a_d1(1, 1).row(q).transpose();   // A_{2,2}
        const Eigen::Matrix<T, 3, 1> A1_d11 = ev.a_d2(0, 0, 0).row(q).transpose();   // A_{1,11}
        const Eigen::Matrix<T, 3, 1> A1_d12 = ev.a_d2(0, 0, 1).row(q).transpose();   // A_{1,12}
        const Eigen::Matrix<T, 3, 1> A1_d22 = ev.a_d2(0, 1, 1).row(q).transpose();   // A_{1,22}
        const Eigen::Matrix<T, 3, 1> A2_d22 = ev.a_d2(1, 1, 1).row(q).transpose();   // A_{2,22}

        // Partial ∂_γ B_{αβ} = A_{αβγ}·A_3 + A_{αβ}·A_{3,γ}.
        const T B11_d1 = A1_d11.dot(A3) + A1_d1.dot(A3_d1);
        const T B11_d2 = A1_d12.dot(A3) + A1_d1.dot(A3_d2);
        const T B12_d1 = A1_d12.dot(A3) + A1_d2.dot(A3_d1);
        const T B12_d2 = A1_d22.dot(A3) + A1_d2.dot(A3_d2);
        const T B22_d1 = A1_d22.dot(A3) + A2_d2.dot(A3_d1);
        const T B22_d2 = A2_d22.dot(A3) + A2_d2.dot(A3_d2);

        // Codazzi-symmetric covariant derivative
        //   B_{αβ|γ} = ∂_γ B_{αβ} − Γ^δ_{αγ}B_{δβ} − Γ^δ_{βγ}B_{αδ}.
        const T B11_cov1 = B11_d1 - T(2) * (G1_11 * B11 + G2_11 * B12);
        const T B11_cov2 = B11_d2 - T(2) * (G1_12 * B11 + G2_12 * B12);
        const T B12_cov1 = B12_d1 - (G1_11 * B12 + G2_11 * B22) - (G1_12 * B11 + G2_12 * B12);
        const T B12_cov2 = B12_d2 - (G1_12 * B12 + G2_12 * B22) - (G1_22 * B11 + G2_22 * B12);
        const T B22_cov1 = B22_d1 - T(2) * (G1_12 * B12 + G2_12 * B22);
        const T B22_cov2 = B22_d2 - T(2) * (G1_22 * B12 + G2_22 * B22);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i = slab0(i);
            const T N_u = slab1(i * 2 + 0);
            const T N_v = slab1(i * 2 + 1);

            // Primal DOF indices
            const Index c_u1 = 4 * i + 0;
            const Index c_u2 = 4 * i + 1;
            const Index c_wb = 4 * i + 2;
            const Index c_ps = 4 * i + 3;

            // w = w_b - Kb/Ks \nabla^2 w_b
            const T box = N_i - ratio * ev.L(i, q);

            // Covariant Hessian H_{αβ} of the scalar potential, and covariant
            // gradient D_{λαβ} = ∂(u_{α|β})/∂u^λ of the in-plane field.
            const T H11 = ev.H(i, 0, 0, q), H12 = ev.H(i, 0, 1, q), H22 = ev.H(i, 1, 1, q);
            const T D000 = ev.D(i,0,0,0,q), D001 = ev.D(i,0,0,1,q),
                    D010 = ev.D(i,0,1,0,q), D011 = ev.D(i,0,1,1,q),
                    D100 = ev.D(i,1,0,0,q), D101 = ev.D(i,1,0,1,q),
                    D110 = ev.D(i,1,1,0,q), D111 = ev.D(i,1,1,1,q);

            // --- Membrane: ε_{αβ} = ½(u_{α|β}+u_{β|α}) − B_{αβ} □(w_b) --------------

            // In-plane u^λ via the covariant-gradient kernel u_{α|β} = D_{λαβ}.
            B(8*q + 0, c_u1) =  D000;
            B(8*q + 0, c_u2) =  D100;
            B(8*q + 1, c_u1) =  D011;
            B(8*q + 1, c_u2) =  D111;
            B(8*q + 2, c_u1) =  D001 + D010;
            B(8*q + 2, c_u2) =  D101 + D110;

            // Scalar Bending Potential: w_b

            // ε_11 += (A_1 · A_{3,1}) * (w_b - Kb/Ks Δw_b)
            B(8*q + 0, c_wb) = -B11 * box;
            B(8*q + 1, c_wb) = -B22 * box;
            B(8*q + 2, c_wb) = -T(2) * B12 * box;

            // --- Bending ------------------------------------------------------------

            // Bending Potential: w_b

            // κ_{αβ} += w_{b,αβ} - (B^2)_{αβ} * (w_b - Kb/Ks Δw_b)
            B(8*q + 3, c_wb) =  H11 - B2_11 * box;
            B(8*q + 4, c_wb) =  H22 - B2_22 * box;
            B(8*q + 5, c_wb) =  T(2) * H12 - T(2) * B2_12 * box;

            // Twist Potential: ψ

            // κ_{αβ} += e_α^λ * ψ_{,λβ} + e_β^λ * ψ_{,λα}
            B(8*q + 3, c_ps) = - (e1_1 * H11 + e1_2 * H12);
            B(8*q + 4, c_ps) = - (e2_1 * H12 + e2_2 * H22);
            B(8*q + 5, c_ps) = - (e1_1 * H12 + e1_2 * H22 + e2_1 * H11 + e2_2 * H12);

            // Contravariant Membrane Displacements: u^λ

            // κ_{αβ} += B_α^μ u_{μ|β} + B_β^μ u_{μ|α}  (coeff 1 on the symmetric sum),
            //   B_α^μ u_{μ|β} = B_α^1 D(i,λ,0,β) + B_α^2 D(i,λ,1,β),  B_α^μ = Bmix(μ,α).
            B(8*q + 3, c_u1) =  T(2) * (Bmix11 * D000 + Bmix21 * D010);
            B(8*q + 3, c_u2) =  T(2) * (Bmix11 * D100 + Bmix21 * D110);
            B(8*q + 4, c_u1) =  T(2) * (Bmix12 * D001 + Bmix22 * D011);
            B(8*q + 4, c_u2) =  T(2) * (Bmix12 * D101 + Bmix22 * D111);
            B(8*q + 5, c_u1) =  T(2) * (Bmix11 * D001 + Bmix21 * D011 + Bmix12 * D000 + Bmix22 * D010);
            B(8*q + 5, c_u2) =  T(2) * (Bmix11 * D101 + Bmix21 * D111 + Bmix12 * D100 + Bmix22 * D110);

            // κ_{αβ} += ½(∇_β B_{αλ} + ∇_α B_{βλ}) u^λ   (Codazzi ∇B; relocated shear coupling)
            B(8*q + 3, c_u1) += N_i * B11_cov1;
            B(8*q + 3, c_u2) += N_i * B12_cov1;
            B(8*q + 4, c_u1) += N_i * B12_cov2;
            B(8*q + 4, c_u2) += N_i * B22_cov2;
            B(8*q + 5, c_u1) += N_i * (B11_cov2 + B12_cov1);
            B(8*q + 5, c_u2) += N_i * (B12_cov2 + B22_cov1);

            // --- Transverse Shear Strain --------------------------------------------

            // \gamma_1 = -(K_b/K_s)(Δ_g w_b)_{,1} + ( e_1^1 ψ_{,1} + e_1^2 ψ_{,2} )
            B(8*q + 6, c_wb) = -ratio * ev.P(i, 0, q);
            B(8*q + 6, c_ps) =  e1_1 * N_u + e1_2 * N_v;

            // \gamma_2 = -(K_b/K_s)(Δ_g w_b)_{,2} + ( e_2^1 ψ_{,1} + e_2^2 ψ_{,2} )
            B(8*q + 7, c_wb) = -ratio * ev.P(i, 1, q);
            B(8*q + 7, c_ps) =  e2_1 * N_u + e2_2 * N_v;
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlin4p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    // D = blockdiag(D_m, D_b, D_s), identical to the 5-parameter shell.
    const Eigen::Matrix<T, 3, 1> g_inv_q = g_inv_voigt(ev, q);

    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(8, 8);
    D.template block<3, 3>(0, 0) = material_->membrane_voigt(g_inv_q);
    D.template block<3, 3>(3, 3) = material_->bending_voigt(g_inv_q);
    D.template block<2, 2>(6, 6) = material_->shear_voigt(g_inv_q);
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
ShellReissnerMindlin4p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& U = this->N_w_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    U.setZero(3 * Q, 4 * N);

    auto A1v = ev.a(0);   // covariant tangent A_1 (Q × 3)
    auto A2v = ev.a(1);   // covariant tangent A_2

    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.results_[0].col(q);

        // Covariant tangents A_α and the unit normal A_3.
        const Eigen::Matrix<T, 3, 1> a1 = A1v.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a2 = A2v.row(q).transpose();
        Eigen::Matrix<T, 3, 1> A3 = a1.cross(a2);
        A3.normalize();

        for (Index i = 0; i < N; ++i) {
            const T N_i = slab0(i);
            // Recovered transverse □_i = N_i − (K_b/K_s) Δ_g N_i = N_i − ratio·L_i.
            const T box = N_i - ratio * ev.L(i, q);

            // Physical displacement u = u¹ A_1 + u² A_2 + □(w_b) A_3 (ψ does no work).
            for (Index r = 0; r < 3; ++r) {
                U(3 * q + r, 4 * i + 0) = a1(r) * N_i;
                U(3 * q + r, 4 * i + 1) = a2(r) * N_i;
                U(3 * q + r, 4 * i + 2) = A3(r) * box;
            }
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlin4p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Director tilt w_α = −w_{b,α} + ε_α^β ψ_{,β}.
    Matrix<T>& Nphi = this->N_phi_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Nphi.setZero(2 * Q, 4 * N);

    for (Index q = 0; q < Q; ++q) {
        auto slab1 = ev.results_[1].col(q);

        const T A11 = ev.g(0, 0)(q);
        const T A12 = ev.g(0, 1)(q);
        const T A22 = ev.g(1, 1)(q);
        const T invJ = T(1) / ev.jac(q);
        const T e1_1 = -A12 * invJ, e1_2 = A11 * invJ;
        const T e2_1 = -A22 * invJ, e2_2 = A12 * invJ;

        for (Index i = 0; i < N; ++i) {
            const T N_u = slab1(i * 2 + 0);
            const T N_v = slab1(i * 2 + 1);
            Nphi(2*q,     4*i + 2) = -N_u;
            Nphi(2*q,     4*i + 3) =  e1_1 * N_u + e1_2 * N_v;
            Nphi(2*q + 1, 4*i + 2) = -N_v;
            Nphi(2*q + 1, 4*i + 3) =  e2_1 * N_u + e2_2 * N_v;
        }
    }
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin4p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin4p<float>;
#endif

} // namespace pyck
