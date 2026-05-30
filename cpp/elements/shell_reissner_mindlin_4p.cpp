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
    auto aux = compute_laplace_grad_aux(ev);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(8 * Q, 4 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = ev.results_[0].col(q);
        auto slab1 = ev.results_[1].col(q);
        auto slab2 = ev.results_[2].col(q);
        auto slab3 = ev.results_[3].col(q);

        const T G1_11 = ev.Gamma(0, 0, 0)(q);
        const T G1_12 = ev.Gamma(0, 0, 1)(q);
        const T G1_22 = ev.Gamma(0, 1, 1)(q);
        const T G2_11 = ev.Gamma(1, 0, 0)(q);
        const T G2_12 = ev.Gamma(1, 0, 1)(q);
        const T G2_22 = ev.Gamma(1, 1, 1)(q);

        // Contravariant metric (for the Laplace–Beltrami operator) and covariant
        // metric + Jacobian (for the surface permutation ε_α^β).
        const T gi11 = ev.g_inv(0, 0)(q);
        const T gi12 = ev.g_inv(0, 1)(q);
        const T gi22 = ev.g_inv(1, 1)(q);
        const T gc11 = ev.g(0, 0)(q);
        const T gc12 = ev.g(0, 1)(q);
        const T gc22 = ev.g(1, 1)(q);
        const T invJ = T(1) / ev.jac(q);

        // Mixed surface permutation ε_α^β = g_{αμ} ϵ^{μβ} / √g, reducing to
        // ε_1^2 = +1, ε_2^1 = −1 on the orthonormal lamina.
        const T e1_1 = -gc12 * invJ, e1_2 = gc11 * invJ;
        const T e2_1 = -gc22 * invJ, e2_2 = gc12 * invJ;

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
        const Eigen::Matrix<T, 3, 1> A3   = ev.n.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> A3_1 = ev.n_d1(0).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> A3_2 = ev.n_d1(1).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a11 = ev.a_d1(0, 0).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a12 = ev.a_d1(0, 1).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a22 = ev.a_d1(1, 1).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a111 = ev.a_d2(0, 0, 0).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a112 = ev.a_d2(0, 0, 1).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a122 = ev.a_d2(0, 1, 1).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a222 = ev.a_d2(1, 1, 1).row(q).transpose();

        // ∂_γ B_{αβ}
        const T dB11_1 = a111.dot(A3) + a11.dot(A3_1);
        const T dB11_2 = a112.dot(A3) + a11.dot(A3_2);
        const T dB12_1 = a112.dot(A3) + a12.dot(A3_1);
        const T dB12_2 = a122.dot(A3) + a12.dot(A3_2);
        const T dB22_1 = a122.dot(A3) + a22.dot(A3_1);
        const T dB22_2 = a222.dot(A3) + a22.dot(A3_2);

        // B_{αβ|γ} = ∂_γ B_{αβ} − Γ^δ_{αγ}B_{δβ} − Γ^δ_{βγ}B_{αδ}  (Codazzi-symmetric).
        const T Bc11_1 = dB11_1 - T(2) * (G1_11 * B11 + G2_11 * B12);
        const T Bc11_2 = dB11_2 - T(2) * (G1_12 * B11 + G2_12 * B12);
        const T Bc12_1 = dB12_1 - (G1_11 * B12 + G2_11 * B22) - (G1_12 * B11 + G2_12 * B12);
        const T Bc12_2 = dB12_2 - (G1_12 * B12 + G2_12 * B22) - (G1_22 * B11 + G2_22 * B12);
        const T Bc22_1 = dB22_1 - T(2) * (G1_12 * B12 + G2_12 * B22);
        const T Bc22_2 = dB22_2 - T(2) * (G1_22 * B12 + G2_22 * B22);

        // Raise the contracted index: (B_α^γ)_{|β} = A^{γμ} B_{αμ|β}.  DB[g][a][b].
        const T DB_1_1_1 = gi11 * Bc11_1 + gi12 * Bc12_1;  // (B_1^1)_{|1}
        const T DB_2_1_1 = gi12 * Bc11_1 + gi22 * Bc12_1;  // (B_1^2)_{|1}
        const T DB_1_2_2 = gi11 * Bc12_2 + gi12 * Bc22_2;  // (B_2^1)_{|2}
        const T DB_2_2_2 = gi12 * Bc12_2 + gi22 * Bc22_2;  // (B_2^2)_{|2}
        const T DB_1_1_2 = gi11 * Bc11_2 + gi12 * Bc12_2;  // (B_1^1)_{|2}
        const T DB_2_1_2 = gi12 * Bc11_2 + gi22 * Bc12_2;  // (B_1^2)_{|2}
        const T DB_1_2_1 = gi11 * Bc12_1 + gi12 * Bc22_1;  // (B_2^1)_{|1}
        const T DB_2_2_1 = gi12 * Bc12_1 + gi22 * Bc22_1;  // (B_2^2)_{|1}

        // Symmetrised ∇B coefficient of û_γ in each bending row:
        //   κ_{αβ} ⊃ ½((B_α^γ)_{|β} + (B_β^γ)_{|α}) u_γ.
        const T gradB_11_u1 = DB_1_1_1;                 // κ_11, u_1
        const T gradB_11_u2 = DB_2_1_1;                 // κ_11, u_2
        const T gradB_22_u1 = DB_1_2_2;                 // κ_22, u_1
        const T gradB_22_u2 = DB_2_2_2;                 // κ_22, u_2
        const T gradB_12_u1 = DB_1_1_2 + DB_1_2_1;      // 2κ_12, u_1
        const T gradB_12_u2 = DB_2_1_2 + DB_2_2_1;      // 2κ_12, u_2

        const T G11_d1 = aux.G_inv_d[0][0][0](q);
        const T G12_d1 = aux.G_inv_d[0][1][0](q);
        const T G22_d1 = aux.G_inv_d[1][1][0](q);
        const T G11_d2 = aux.G_inv_d[0][0][1](q);
        const T G12_d2 = aux.G_inv_d[0][1][1](q);
        const T G22_d2 = aux.G_inv_d[1][1][1](q);
        const T c1 = aux.c[0](q),    c2 = aux.c[1](q);
        const T c1_d1 = aux.c_d[0][0](q), c2_d1 = aux.c_d[1][0](q);
        const T c1_d2 = aux.c_d[0][1](q), c2_d2 = aux.c_d[1][1](q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i    = slab0(i);
            const T N_u    = slab1(i * 2 + 0);
            const T N_v    = slab1(i * 2 + 1);
            const T N_uu   = slab2(i * 3 + 0);
            const T N_vv   = slab2(i * 3 + 1);
            const T N_uv   = slab2(i * 3 + 2);
            const T N_uuu  = slab3(i * 4 + 0);
            const T N_uuv  = slab3(i * 4 + 1);
            const T N_uvv  = slab3(i * 4 + 2);
            const T N_vvv  = slab3(i * 4 + 3);

            // Covariant Hessian N_{i|αβ}.
            const T N11 = N_uu - G1_11 * N_u - G2_11 * N_v;
            const T N12 = N_uv - G1_12 * N_u - G2_12 * N_v;
            const T N22 = N_vv - G1_22 * N_u - G2_22 * N_v;

            // Recovered transverse shape □_i = N_i − (K_b/K_s) Δ_g N_i.
            const T lapN = gi11 * N11 + T(2) * gi12 * N12 + gi22 * N22;
            const T box  = N_i - ratio * lapN;

            // Surface gradient of the Laplace–Beltrami, (K_b/K_s)(Δ_g N_i)_{,α}.
            const T lap_1 = ratio * (
                  G11_d1 * N_uu  + T(2)*G12_d1 * N_uv + G22_d1 * N_vv
                + gi11   * N_uuu + T(2)*gi12   * N_uuv + gi22   * N_uvv
                - c1_d1  * N_u   - c2_d1       * N_v
                - c1     * N_uu  - c2          * N_uv);
            const T lap_2 = ratio * (
                  G11_d2 * N_uu  + T(2)*G12_d2 * N_uv + G22_d2 * N_vv
                + gi11   * N_uuv + T(2)*gi12   * N_uvv + gi22   * N_vvv
                - c1_d2  * N_u   - c2_d2       * N_v
                - c1     * N_uv  - c2          * N_vv);

            const Index c_u1 = 4 * i + 0;
            const Index c_u2 = 4 * i + 1;
            const Index c_wb = 4 * i + 2;
            const Index c_ps = 4 * i + 3;

            // --- Membrane Strain \varepsilon_{ab} -----------------------------------

            // \varepsilon_{11} = u_{1,1} - \Gamma^1_{11} u_1 
            //                            - \Gamma^2_{11} u_2 - B_{11} u^3
            B(8*q + 0, c_u1) =  N_u - G1_11 * N_i;
            B(8*q + 0, c_u2) =      - G2_11 * N_i;
            B(8*q + 0, c_wb) = -B11 * box;

            // \varepsilon_{22} = u_{2,2} - \Gamma^1_{22} u_1 
            //                            - \Gamma^2_{22} u_2 - B_{22} u^3
            B(8*q + 1, c_u1) =      - G1_22 * N_i;
            B(8*q + 1, c_u2) =  N_v - G2_22 * N_i;
            B(8*q + 1, c_wb) = -B22 * box;

            // 2 \varepsilon_{12} = u_{1,2} + u_{2,1} - 2 \Gamma^1_{12} u_1 - 
            //                                          2 \Gamma^2_{12} u_2 - 2 B_{12} u^3
            B(8*q + 2, c_u1) =  N_v - T(2) * G1_12 * N_i;
            B(8*q + 2, c_u2) =  N_u - T(2) * G2_12 * N_i;
            B(8*q + 2, c_wb) = -T(2) * B12 * box;

            // --- Bending Strain \kappa_{ab} -----------------------------------------
            
            // \kappa_{11} = w_{b|11} - (B^2)_{11} u^3                      // w_b
            //             - ( e_1^1 * \psi_{|11} + e_1^2 \psi_{|21} )      // psi
            //             + ( B_1^1 * u_{1|1} + B_1^2 * u_{2|1} )          // u1, u2
            //             + ( (B_1^1)_{|1} u_1 + (B_1^2)_{|1} u_2 )        // ∇B (deformed director)
            B(8*q + 3, c_wb) =  N11 - B2_11 * box;
            B(8*q + 3, c_ps) = - (e1_1 * N11 + e1_2 * N12);
            B(8*q + 3, c_u1) = - (Bmix11 * G1_11 + Bmix21 * G1_12) * N_i + Bmix11 * N_u + gradB_11_u1 * N_i;
            B(8*q + 3, c_u2) = - (Bmix11 * G2_11 + Bmix21 * G2_12) * N_i + Bmix21 * N_u + gradB_11_u2 * N_i;

            // \kappa_{22} = w_{b|22} - (B^2)_{22} u^3                      // w_b
            //             - ( e_2^1 * \psi_{|12} + e_2^2 \psi_{|22} )      // psi
            //             + ( B_2^1 * u_{1|1} + B_2^2 * u_{2|2} )          // u1, u2
            //             + ( (B_2^1)_{|2} u_1 + (B_2^2)_{|2} u_2 )        // ∇B (deformed director)
            B(8*q + 4, c_wb) =  N22 - B2_22 * box;
            B(8*q + 4, c_ps) = - (e2_1 * N12 + e2_2 * N22);
            B(8*q + 4, c_u1) = - (Bmix12 * G1_12 + Bmix22 * G1_22) * N_i + Bmix12 * N_v + gradB_22_u1 * N_i;
            B(8*q + 4, c_u2) = - (Bmix12 * G2_12 + Bmix22 * G2_22) * N_i + Bmix22 * N_v + gradB_22_u2 * N_i;

            // 2 \kappa_{12} = 2 w_{b|12} - 2 (B^2)_{12} u^3                // w_b
            //               - ( e_1^1 \psi_{|12} + e_1^2 * \psi_{|22} +    // psi
            //                   e_2^1 \psi_{|11} + e_2^2 * \psi_{|21} )
            //               + ( B_1^1 u_{1|2} + B_1^2 u_{2|2} +            // u1, u2
            //                   B_2^1 u_{1|1} + B_2^2 u_{2|1} )
            //               + ( ((B_1^γ)_{|2}+(B_2^γ)_{|1}) u_γ )          // ∇B (deformed director)
            B(8*q + 5, c_wb) =  T(2) * N12 - T(2) * B2_12 * box;
            B(8*q + 5, c_ps) = - (e1_1 * N12 + e1_2 * N22 + e2_1 * N11 + e2_2 * N12);
            B(8*q + 5, c_u1) =  Bmix11 * N_v + Bmix12 * N_u
                             - (Bmix11 * G1_12 + Bmix21 * G1_22 +
                                Bmix12 * G1_11 + Bmix22 * G1_12) * N_i + gradB_12_u1 * N_i;
            B(8*q + 5, c_u2) =  Bmix21 * N_v + Bmix22 * N_u
                             - (Bmix11 * G2_12 + Bmix21 * G2_22 +
                                Bmix12 * G2_11 + Bmix22 * G2_12) * N_i + gradB_12_u2 * N_i;

            // --- Transverse Shear Strain --------------------------------------------
            // Deformed-director referencing: θ_α carries −B_α^γ u_γ, which cancels
            // the membrane–curvature coupling B_{αγ}u^γ here, leaving only the
            // recovered shear deflection and the twist. Shear-locking-free by
            // construction; the curvature coupling reappears (consistently, with
            // its ∇B partner) in the bending block above.

            // \gamma_1 = -(K_b/K_s)(Δ_g w_b)_{,1} + ( e_1^1 ψ_{,1} + e_1^2 ψ_{,2} )
            B(8*q + 6, c_wb) = -lap_1;
            B(8*q + 6, c_ps) =  e1_1 * N_u + e1_2 * N_v;

            // \gamma_2 = -(K_b/K_s)(Δ_g w_b)_{,2} + ( e_2^1 ψ_{,1} + e_2^2 ψ_{,2} )
            B(8*q + 7, c_wb) = -lap_2;
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
        auto slab1 = ev.results_[1].col(q);
        auto slab2 = ev.results_[2].col(q);

        const T gi11 = ev.g_inv(0, 0)(q);
        const T gi12 = ev.g_inv(0, 1)(q);
        const T gi22 = ev.g_inv(1, 1)(q);
        const T G1_11 = ev.Gamma(0, 0, 0)(q);
        const T G1_12 = ev.Gamma(0, 0, 1)(q);
        const T G1_22 = ev.Gamma(0, 1, 1)(q);
        const T G2_11 = ev.Gamma(1, 0, 0)(q);
        const T G2_12 = ev.Gamma(1, 0, 1)(q);
        const T G2_22 = ev.Gamma(1, 1, 1)(q);

        // Contravariant tangents A^α = g^{αβ} A_β and the unit normal A_3.
        const Eigen::Matrix<T, 3, 1> a1 = A1v.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a2 = A2v.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> Aup1 = gi11 * a1 + gi12 * a2;
        const Eigen::Matrix<T, 3, 1> Aup2 = gi12 * a1 + gi22 * a2;
        Eigen::Matrix<T, 3, 1> A3 = a1.cross(a2);
        A3.normalize();

        for (Index i = 0; i < N; ++i) {
            const T N_i  = slab0(i);
            const T N_u  = slab1(i * 2 + 0);
            const T N_v  = slab1(i * 2 + 1);
            const T N_uu = slab2(i * 3 + 0);
            const T N_vv = slab2(i * 3 + 1);
            const T N_uv = slab2(i * 3 + 2);

            const T N11 = N_uu - G1_11 * N_u - G2_11 * N_v;
            const T N12 = N_uv - G1_12 * N_u - G2_12 * N_v;
            const T N22 = N_vv - G1_22 * N_u - G2_22 * N_v;

            // Recovered transverse □_i = N_i − (K_b/K_s) Δ_g N_i.
            const T box = N_i - ratio * (gi11 * N11 + T(2) * gi12 * N12 + gi22 * N22);

            // Physical displacement u = u₁ A^1 + u₂ A^2 + □(w_b) A_3 (ψ does no work).
            for (Index r = 0; r < 3; ++r) {
                U(3 * q + r, 4 * i + 0) = Aup1(r) * N_i;
                U(3 * q + r, 4 * i + 1) = Aup2(r) * N_i;
                U(3 * q + r, 4 * i + 2) = A3(r)  * box;
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

        const T gc11 = ev.g(0, 0)(q);
        const T gc12 = ev.g(0, 1)(q);
        const T gc22 = ev.g(1, 1)(q);
        const T invJ = T(1) / ev.jac(q);
        const T e1_1 = -gc12 * invJ, e1_2 = gc11 * invJ;
        const T e2_1 = -gc22 * invJ, e2_2 = gc12 * invJ;

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
