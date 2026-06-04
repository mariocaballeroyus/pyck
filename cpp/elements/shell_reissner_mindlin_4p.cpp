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
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    // Reset strain matrix values
    Matrix<T>& B_voigt = this->B_voigt_;
    B_voigt.setZero(8 * Q, 4 * N);

    // Bending-stiffness ratio for kinematic relation
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    for (Index q = 0; q < Q; ++q)
    {
        // --- Precomputed geometric primitives ---------------------------------------

        // Second fundamental form
        const auto B = ev.curvature(q);
        const T B11 = B(0, 0), B22 = B(1, 1), B12 = B(0, 1);
        // Contravariant metric
        const auto Ainv = ev.metric_inv(q);
        const T A11inv = Ainv(0, 0), A22inv = Ainv(1, 1), A12inv = Ainv(0, 1);
        // Unit normal
        const Vector3<T> A3 = ev.normal(q);
        // Unit normal derivatives A_{3,β}
        const auto nd = ev.normal_deriv(q);
        const Vector3<T> A3_d1 = nd(0), A3_d2 = nd(1);
        // Covariant basis 1st derivatives
        const auto A_d = ev.cov_basis_d(q);
        const Vector3<T> A1_d1 = A_d(0, 0), A1_d2 = A_d(0, 1),
                         A2_d2 = A_d(1, 1);
        // Covariant basis 2nd derivatives
        const auto A_dd = ev.cov_basis_dd(q);
        const Vector3<T> A1_d11 = A_dd(0, 0, 0), A1_d12 = A_dd(0, 0, 1),
                         A1_d22 = A_dd(0, 1, 1), A2_d22 = A_dd(1, 1, 1);
        // Christoffel symbols of the 1st kind
        const auto Gam = ev.christoffel(q);
        const T G1_11 = Gam(0, 0, 0), G1_12 = Gam(0, 0, 1), G1_22 = Gam(0, 1, 1);
        const T G2_11 = Gam(1, 0, 0), G2_12 = Gam(1, 0, 1), G2_22 = Gam(1, 1, 1);

        // --- Formulation-specific primitives ----------------------------------------

        // Shape operator B^α_β = A^{αγ} B_{γβ}
        const T Bmix11 = A11inv * B11 + A12inv * B12;
        const T Bmix12 = A11inv * B12 + A12inv * B22;
        const T Bmix21 = A12inv * B11 + A22inv * B12;
        const T Bmix22 = A12inv * B12 + A22inv * B22;
        // Third fundamental form (B²)_{αβ} = B_{αμ} B^μ_β = A_{3,α}·A_{3,β}
        const T B2_11 = B11 * Bmix11 + B12 * Bmix21;
        const T B2_22 = B12 * Bmix12 + B22 * Bmix22;
        const T B2_12 = B11 * Bmix12 + B12 * Bmix22;
        // Second fundamental form derivatives
        const T B11_d1 = A1_d11.dot(A3) + A1_d1.dot(A3_d1);
        const T B11_d2 = A1_d12.dot(A3) + A1_d1.dot(A3_d2);
        const T B12_d1 = A1_d12.dot(A3) + A1_d2.dot(A3_d1);
        const T B12_d2 = A1_d22.dot(A3) + A1_d2.dot(A3_d2);
        const T B22_d1 = A1_d22.dot(A3) + A2_d2.dot(A3_d1);
        const T B22_d2 = A2_d22.dot(A3) + A2_d2.dot(A3_d2);
        // Codazzi-symmetric covariant derivative
        const T B11_cov1 = B11_d1 - T(2) * (G1_11 * B11 + G2_11 * B12);
        const T B11_cov2 = B11_d2 - T(2) * (G1_12 * B11 + G2_12 * B12);
        const T B12_cov1 = B12_d1 - (G1_11 * B12 + G2_11 * B22) 
                                  - (G1_12 * B11 + G2_12 * B12);
        const T B12_cov2 = B12_d2 - (G1_12 * B12 + G2_12 * B22) 
                                  - (G1_22 * B11 + G2_22 * B12);
        const T B22_cov1 = B22_d1 - T(2) * (G1_12 * B12 + G2_12 * B22);
        const T B22_cov2 = B22_d2 - T(2) * (G1_22 * B12 + G2_22 * B22);

        // --- Shape operators --------------------------------------------------------

        const auto Nf = ev.N(q);
        const operators::CovariantGradient<T, 2>       vgrad{ev, q};
        const operators::Curl<T, 2>                    curl {ev, q};
        const operators::CovariantHessian<T, 2>        hess {ev, q};
        const operators::LaplaceBeltrami<T, 2>         lapb {ev, q};
        const operators::CurlGradient<T, 2>            curlgrad{ev, q};
        const operators::LaplaceBeltramiGradient<T, 2> lgrad{ev, q};

        for (Index i = 0; i < N; ++i) {
            // Primal DOF indices
            const Index idx_u1 = 4 * i + 0;
            const Index idx_u2 = 4 * i + 1;
            const Index idx_wb = 4 * i + 2;
            const Index idx_psi = 4 * i + 3;

            const T Ni = Nf(i);
            const T H11 = hess(i, 0, 0), H12 = hess(i, 0, 1), H22 = hess(i, 1, 1);
            const T D000 = vgrad(i,0,0,0), D001 = vgrad(i,0,0,1),
                    D010 = vgrad(i,0,1,0), D011 = vgrad(i,0,1,1),
                    D100 = vgrad(i,1,0,0), D101 = vgrad(i,1,0,1),
                    D110 = vgrad(i,1,1,0), D111 = vgrad(i,1,1,1);

            // Total transverse displacement w = w_b - Kb/Ks \nabla^2 w_b
            const T box = Ni - ratio * lapb(i);

            // --- Membrane Strain ----------------------------------------------------

            // Symmetric contravariant gradient of the in-plane displacements 
            // ε_{αβ} += (1/2) * ( u_{α|β} + u_{β|α} )
            B_voigt(8*q + 0, idx_u1) =  D000;
            B_voigt(8*q + 0, idx_u2) =  D100;
            B_voigt(8*q + 1, idx_u1) =  D011;
            B_voigt(8*q + 1, idx_u2) =  D111;
            B_voigt(8*q + 2, idx_u1) =  D001 + D010;
            B_voigt(8*q + 2, idx_u2) =  D101 + D110;

            // Membrane coupling of the transverse displacement (curvature-driven)
            // ε_{αβ} += B_{αβ} * ( w_b - Kb/Ks * Δ w_b )
            B_voigt(8*q + 0, idx_wb) = -B11 * box;
            B_voigt(8*q + 1, idx_wb) = -B22 * box;
            B_voigt(8*q + 2, idx_wb) = -T(2) * B12 * box;

            // --- Bending Strain -----------------------------------------------------

            // Symmetric covariant Hessian of the bending potential with curvature 
            // coupling correction
            // κ_{αβ} += w_{b|αβ} - (B^2)_{αβ} * (w_b - Kb/Ks Δ w_b)
            B_voigt(8*q + 3, idx_wb) =  hess(i, 0, 0) - B2_11 * box;
            B_voigt(8*q + 4, idx_wb) =  hess(i, 1, 1) - B2_22 * box;
            B_voigt(8*q + 5, idx_wb) =  T(2) * hess(i, 0, 1) - T(2) * B2_12 * box;

            // Covariant gradient of the curl of the twist potential
            // κ_{αβ} += -(1/2)( ε_α^δ ψ_{|δβ} + ε_β^δ ψ_{|δα} ) = -(curl ψ)_{(α|β)}
            B_voigt(8*q + 3, idx_psi) = -curlgrad(i, 0, 0);
            B_voigt(8*q + 4, idx_psi) = -curlgrad(i, 1, 1);
            B_voigt(8*q + 5, idx_psi) = -(curlgrad(i, 0, 1) + curlgrad(i, 1, 0));

            // Bending coupling of the symmetric contravariant gradient of the 
            // in-plane displacements
            // κ_{αβ} += B_α^μ * u_{μ|β} + B_β^μ * u_{μ|α}
            B_voigt(8*q + 3, idx_u1) =  T(2) * (Bmix11 * D000 + Bmix21 * D010);
            B_voigt(8*q + 3, idx_u2) =  T(2) * (Bmix11 * D100 + Bmix21 * D110);
            B_voigt(8*q + 4, idx_u1) =  T(2) * (Bmix12 * D001 + Bmix22 * D011);
            B_voigt(8*q + 4, idx_u2) =  T(2) * (Bmix12 * D101 + Bmix22 * D111);
            B_voigt(8*q + 5, idx_u1) =  T(2) * (Bmix11 * D001 + Bmix21 * D011 + 
                                        Bmix12 * D000 + Bmix22 * D010);
            B_voigt(8*q + 5, idx_u2) =  T(2) * (Bmix11 * D101 + Bmix21 * D111 + 
                                        Bmix12 * D100 + Bmix22 * D110);

            // Covariant derivative of the curvature (Codazzi coupling)
            // κ_{αβ} += B^μ_{α|β} * u_μ (symmetric)
            B_voigt(8*q + 3, idx_u1) += Ni * B11_cov1;
            B_voigt(8*q + 3, idx_u2) += Ni * B12_cov1;
            B_voigt(8*q + 4, idx_u1) += Ni * B12_cov2;
            B_voigt(8*q + 4, idx_u2) += Ni * B22_cov2;
            B_voigt(8*q + 5, idx_u1) += Ni * (B11_cov2 + B12_cov1);
            B_voigt(8*q + 5, idx_u2) += Ni * (B12_cov2 + B22_cov1);

            // --- Transverse Shear Strain --------------------------------------------

            // Scaled gradient of the Laplacian of the bending potential
            // γ_{α} = -( Kb/Ks Δ w_b )_{,α}
            B_voigt(8*q + 6, idx_wb) = -ratio * lgrad(i, 0);
            B_voigt(8*q + 7, idx_wb) = -ratio * lgrad(i, 1);

            // Curl of the twist potential
            // γ_{α} += ε_α^β ψ_{,β}
            B_voigt(8*q + 6, idx_psi) =  curl(i, 0);
            B_voigt(8*q + 7, idx_psi) =  curl(i, 1);
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlin4p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
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
ShellReissnerMindlin4p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& U = this->N_w_;
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    U.setZero(3 * Q, 4 * N);

    for (Index q = 0; q < Q; ++q) {
        const operators::LaplaceBeltrami<T, 2> lapb{ev, q};

        // Covariant tangents A_α and the unit normal A_3.
        const auto A = ev.cov_basis(q);
        const Vector3<T> a1 = A(0);
        const Vector3<T> a2 = A(1);
        Vector3<T> A3 = a1.cross(a2);
        A3.normalize();
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);
            // Recovered transverse □_i = Ni − (K_b/K_s) Δ_g Ni = Ni − ratio·L_i.
            const T box = Ni - ratio * lapb(i);

            // Physical displacement u = u¹ A_1 + u² A_2 + □(w_b) A_3 (ψ does no work).
            for (Index r = 0; r < 3; ++r) {
                U(3 * q + r, 4 * i + 0) = a1(r) * Ni;
                U(3 * q + r, 4 * i + 1) = a2(r) * Ni;
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
    Matrix<T>& Nphi = this->N_phi_;
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    Nphi.setZero(2 * Q, 4 * N);

    for (Index q = 0; q < Q; ++q) {
        const operators::Curl<T, 2> curl{ev, q};
        const auto G = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T N_u = G(i, 0), N_v = G(i, 1);
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
