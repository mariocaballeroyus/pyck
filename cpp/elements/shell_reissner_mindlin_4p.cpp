#include "shell_reissner_mindlin_4p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"
#include "../operators/laplace_beltrami.hpp"
#include "../operators/covariant_gradient.hpp"
#include "../operators/laplace_beltrami_gradient.hpp"
#include "../operators/curl.hpp"
#include "../operators/curl_gradient.hpp"

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

        const operators::CovariantGradient<T, 2> vgrad{ev.results_, ev.position_data, q};
        const operators::Curl<T, 2>              curl {ev.results_, ev.g_data, ev.jac, q};
        const operators::CovariantHessian<T, 2>  hess {ev.results_, ev.position_data, ev.g_inv_data, q};
        const operators::LaplaceBeltrami<T, 2>   lapb {ev.results_, ev.position_data, ev.g_inv_data, q};
        const operators::CurlGradient<T, 2>      curlgrad{
            ev.results_, ev.position_data, ev.g_data, ev.g_inv_data, ev.jac, q};
        const operators::LaplaceBeltramiGradient<T, 2> lgrad{
            ev.results_, ev.position_data, ev.g_inv_data, q};

        // Second fundamental form B_{αβ} = b_{αβ} (cached) and shape operator
        // B^α_β = g^{αγ} B_{γβ}, raised in place (b_{21} = b_{12}).
        const T B11 = ev.b(0, 0)(q), B12 = ev.b(0, 1)(q), B22 = ev.b(1, 1)(q);
        const T gi11 = ev.g_inv(0, 0)(q), gi12 = ev.g_inv(0, 1)(q), gi22 = ev.g_inv(1, 1)(q);
        const T Bmix11 = gi11 * B11 + gi12 * B12;  // B^1_1
        const T Bmix12 = gi11 * B12 + gi12 * B22;  // B^1_2
        const T Bmix21 = gi12 * B11 + gi22 * B12;  // B^2_1
        const T Bmix22 = gi12 * B12 + gi22 * B22;  // B^2_2

        // (B^2)_{αβ} = B_{αμ} B^μ_β.
        const T B2_11 = B11 * Bmix11 + B12 * Bmix21;
        const T B2_22 = B12 * Bmix12 + B22 * Bmix22;
        const T B2_12 = B11 * Bmix12 + B12 * Bmix22;

        // ----------------------------------------------------------------------------

        const Eigen::Matrix<T, 3, 1> A3 = ev.n.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a1 = ev.a(0).row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a2 = ev.a(1).row(q).transpose();
        // Weingarten: A_{3,β} = −B^α_β A_α, from the cached second fundamental form.
        const Eigen::Matrix<T, 3, 1> A3_d1 = -(Bmix11 * a1 + Bmix21 * a2);   // A_{3,1}
        const Eigen::Matrix<T, 3, 1> A3_d2 = -(Bmix12 * a1 + Bmix22 * a2);   // A_{3,2}
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

        // ----------------------------------------------------------------------------

        // Second-kind Christoffel Γ^k_{ij} = g^{kγ}(A_γ·A_{,ij}). Formed here per qp
        // (the same connection `hess` hoists, but kept explicit: reading hess.Gamma
        // instead perturbs K at ~1e-18 via FMA contraction, breaking bit-exactness).
        auto Gamma2nd = [&](Index k, Index i, Index j) -> T {
            return ev.g_inv(k, 0)(q) * ev.a(0).row(q).dot(ev.a_d1(i, j).row(q))
                 + ev.g_inv(k, 1)(q) * ev.a(1).row(q).dot(ev.a_d1(i, j).row(q));
        };
        const T G1_11 = Gamma2nd(0, 0, 0);
        const T G1_12 = Gamma2nd(0, 0, 1);
        const T G1_22 = Gamma2nd(0, 1, 1);
        const T G2_11 = Gamma2nd(1, 0, 0);
        const T G2_12 = Gamma2nd(1, 0, 1);
        const T G2_22 = Gamma2nd(1, 1, 1);

        // Codazzi-symmetric covariant derivative
        // B_{αβ|γ} = ∂_γ B_{αβ} − Γ^δ_{αγ}B_{δβ} − Γ^δ_{βγ}B_{αδ}.
        const T B11_cov1 = B11_d1 - T(2) * (G1_11 * B11 + G2_11 * B12);
        const T B11_cov2 = B11_d2 - T(2) * (G1_12 * B11 + G2_12 * B12);
        const T B12_cov1 = B12_d1 - (G1_11 * B12 + G2_11 * B22) - (G1_12 * B11 + G2_12 * B12);
        const T B12_cov2 = B12_d2 - (G1_12 * B12 + G2_12 * B22) - (G1_22 * B11 + G2_22 * B12);
        const T B22_cov1 = B22_d1 - T(2) * (G1_12 * B12 + G2_12 * B22);
        const T B22_cov2 = B22_d2 - T(2) * (G1_22 * B12 + G2_22 * B22);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i = slab0(i);

            // Primal DOF indices
            const Index c_u1 = 4 * i + 0;
            const Index c_u2 = 4 * i + 1;
            const Index c_wb = 4 * i + 2;
            const Index c_ps = 4 * i + 3;

            // w = w_b - Kb/Ks \nabla^2 w_b
            const T box = N_i - ratio * lapb(i);

            // Covariant Hessian H_{αβ} of the scalar potential, and covariant
            // gradient D_{λαβ} = ∂(u_{α|β})/∂u^λ of the in-plane field.
            const T H11 = hess(i, 0, 0), H12 = hess(i, 0, 1), H22 = hess(i, 1, 1);
            const T D000 = vgrad(i,0,0,0), D001 = vgrad(i,0,0,1),
                    D010 = vgrad(i,0,1,0), D011 = vgrad(i,0,1,1),
                    D100 = vgrad(i,1,0,0), D101 = vgrad(i,1,0,1),
                    D110 = vgrad(i,1,1,0), D111 = vgrad(i,1,1,1);

            // --- Membrane Strain ----------------------------------------------------

            // Symmetric contravariant gradient of the in-plane displacements 
            // ε_{αβ} += (1/2) * ( u_{α|β} + u_{β|α} )
            B(8*q + 0, c_u1) =  D000;
            B(8*q + 0, c_u2) =  D100;
            B(8*q + 1, c_u1) =  D011;
            B(8*q + 1, c_u2) =  D111;
            B(8*q + 2, c_u1) =  D001 + D010;
            B(8*q + 2, c_u2) =  D101 + D110;

            // Membrane coupling of the transverse displacement (curvature-driven)
            // ε_{αβ} += B_{αβ} * ( w_b - Kb/Ks * Δ w_b )
            B(8*q + 0, c_wb) = -B11 * box;
            B(8*q + 1, c_wb) = -B22 * box;
            B(8*q + 2, c_wb) = -T(2) * B12 * box;

            // --- Bending Strain -----------------------------------------------------

            // Symmetric covariant Hessian of the bending potential with curvature 
            // coupling correction
            // κ_{αβ} += w_{b|αβ} - (B^2)_{αβ} * (w_b - Kb/Ks Δ w_b)
            B(8*q + 3, c_wb) =  hess(i, 0, 0) - B2_11 * box;
            B(8*q + 4, c_wb) =  hess(i, 1, 1) - B2_22 * box;
            B(8*q + 5, c_wb) =  T(2) * hess(i, 0, 1) - T(2) * B2_12 * box;

            // Covariant gradient of the curl of the twist potential
            // κ_{αβ} += -(1/2)( ε_α^δ ψ_{|δβ} + ε_β^δ ψ_{|δα} ) = -(curl ψ)_{(α|β)}
            B(8*q + 3, c_ps) = -curlgrad(i, 0, 0);
            B(8*q + 4, c_ps) = -curlgrad(i, 1, 1);
            B(8*q + 5, c_ps) = -(curlgrad(i, 0, 1) + curlgrad(i, 1, 0));

            // Bending coupling of the symmetric contravariant gradient of the 
            // in-plane displacements
            // κ_{αβ} += B_α^μ * u_{μ|β} + B_β^μ * u_{μ|α}
            B(8*q + 3, c_u1) =  T(2) * (Bmix11 * D000 + Bmix21 * D010);
            B(8*q + 3, c_u2) =  T(2) * (Bmix11 * D100 + Bmix21 * D110);
            B(8*q + 4, c_u1) =  T(2) * (Bmix12 * D001 + Bmix22 * D011);
            B(8*q + 4, c_u2) =  T(2) * (Bmix12 * D101 + Bmix22 * D111);
            B(8*q + 5, c_u1) =  T(2) * (Bmix11 * D001 + Bmix21 * D011 + 
                                        Bmix12 * D000 + Bmix22 * D010);
            B(8*q + 5, c_u2) =  T(2) * (Bmix11 * D101 + Bmix21 * D111 + 
                                        Bmix12 * D100 + Bmix22 * D110);

            // Covariant derivative of the curvature (Codazzi coupling)
            // κ_{αβ} += B^μ_{α|β} * u_μ (symmetric)
            B(8*q + 3, c_u1) += N_i * B11_cov1;
            B(8*q + 3, c_u2) += N_i * B12_cov1;
            B(8*q + 4, c_u1) += N_i * B12_cov2;
            B(8*q + 4, c_u2) += N_i * B22_cov2;
            B(8*q + 5, c_u1) += N_i * (B11_cov2 + B12_cov1);
            B(8*q + 5, c_u2) += N_i * (B12_cov2 + B22_cov1);

            // --- Transverse Shear Strain --------------------------------------------

            // Scaled gradient of the Laplacian of the bending potential
            // γ_{α} = -( Kb/Ks Δ w_b )_{,α}
            B(8*q + 6, c_wb) = -ratio * lgrad(i, 0);
            B(8*q + 7, c_wb) = -ratio * lgrad(i, 1);

            // Curl of the twist potential
            // γ_{α} += ε_α^β ψ_{,β}
            B(8*q + 6, c_ps) =  curl(i, 0);
            B(8*q + 7, c_ps) =  curl(i, 1);
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

        const operators::LaplaceBeltrami<T, 2> lapb{ev.results_, ev.position_data, ev.g_inv_data, q};

        // Covariant tangents A_α and the unit normal A_3.
        const Eigen::Matrix<T, 3, 1> a1 = A1v.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> a2 = A2v.row(q).transpose();
        Eigen::Matrix<T, 3, 1> A3 = a1.cross(a2);
        A3.normalize();

        for (Index i = 0; i < N; ++i) {
            const T N_i = slab0(i);
            // Recovered transverse □_i = N_i − (K_b/K_s) Δ_g N_i = N_i − ratio·L_i.
            const T box = N_i - ratio * lapb(i);

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

        const operators::Curl<T, 2> curl{ev.results_, ev.g_data, ev.jac, q};

        for (Index i = 0; i < N; ++i) {
            const T N_u = slab1(i * 2 + 0);
            const T N_v = slab1(i * 2 + 1);
            Nphi(2*q,     4*i + 2) = -N_u;
            Nphi(2*q,     4*i + 3) =  curl(i, 0);
            Nphi(2*q + 1, 4*i + 2) = -N_v;
            Nphi(2*q + 1, 4*i + 3) =  curl(i, 1);
        }
    }
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin4p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin4p<float>;
#endif

} // namespace pyck
