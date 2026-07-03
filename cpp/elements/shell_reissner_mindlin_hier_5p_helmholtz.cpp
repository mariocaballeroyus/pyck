#include "shell_reissner_mindlin_hier_5p_helmholtz.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"
#include "../operators/curl.hpp"
#include "../operators/curl_gradient.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlinHier5pHelmholtz<T>::ShellReissnerMindlinHier5pHelmholtz(
    Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlinHier5pHelmholtz: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellReissnerMindlinHier5pHelmholtz<T>::membrane_strain_matrix(const ElementValues<T, 2>& ev,
                                                               Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const auto Bc = ev.curvature(q);
        const T B11 = Bc(0, 0), B22 = Bc(1, 1), B12 = Bc(0, 1);
        const auto A = ev.cov_basis(q);
        const Vector3<T> A1 = A(0), A2 = A(1);
        const auto G = ev.dN(q);
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T N_u = G(i, 0), N_v = G(i, 1);
            // Membrane ε_{αβ}: Kirchhoff-Love A_α·v_{,β} on the v_b slots.
            for (Index k = 0; k < 3; ++k) {
                const Index idx = 5 * i + k;
                B(8*q + 0, idx) = A1(k) * N_u;
                B(8*q + 1, idx) = A2(k) * N_v;
                B(8*q + 2, idx) = A1(k) * N_v + A2(k) * N_u;
            }
            // Shear enrichment += −B_{αβ} w_s, with w_s the independent slot-4 field.
            const Index idx_ws = 5 * i + 4;
            const T Ni = Nf(i);
            B(8*q + 0, idx_ws) =        -B11 * Ni;
            B(8*q + 1, idx_ws) =        -B22 * Ni;
            B(8*q + 2, idx_ws) = -T(2) * B12 * Ni;
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pHelmholtz<T>::bending_strain_matrix(const ElementValues<T, 2>& ev,
                                                              Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto nd = ev.normal_deriv(q);
        const Vector3<T> A3_d1 = nd(0), A3_d2 = nd(1);
        // Third fundamental form (B²)_{αβ} = A_{3,α}·A_{3,β}
        const T B2_11 = A3_d1.dot(A3_d1), B2_22 = A3_d2.dot(A3_d2), B2_12 = A3_d1.dot(A3_d2);
        const operators::CovariantHessian<T, 2> hess{ev, q};
        const operators::CurlGradient<T, 2>     curlgrad{ev, q};
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T H11 = hess(i, 0, 0), H22 = hess(i, 1, 1), H12 = hess(i, 0, 1);
            // Bending κ_{αβ}: Kirchhoff-Love −v_{|αβ}·A_3 on the v_b slots.
            for (Index k = 0; k < 3; ++k) {
                const Index idx = 5 * i + k;
                B(8*q + 3, idx) =        -H11 * A3(k);
                B(8*q + 4, idx) =        -H22 * A3(k);
                B(8*q + 5, idx) = -T(2) * H12 * A3(k);
            }
            // Twist potential ψ: κ_{αβ} += (curl ψ)_{(α|β)}
            const Index idx_psi = 5 * i + 3;
            B(8*q + 3, idx_psi) = curlgrad(i, 0, 0);
            B(8*q + 4, idx_psi) = curlgrad(i, 1, 1);
            B(8*q + 5, idx_psi) = curlgrad(i, 0, 1) + curlgrad(i, 1, 0);
            // Shear enrichment += w_s (B²)_{αβ}, with w_s the independent slot-4 field.
            const Index idx_ws = 5 * i + 4;
            const T Ni = Nf(i);
            B(8*q + 3, idx_ws) =        Ni * B2_11;
            B(8*q + 4, idx_ws) =        Ni * B2_22;
            B(8*q + 5, idx_ws) = T(2) * Ni * B2_12;
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pHelmholtz<T>::shear_strain_matrix(const ElementValues<T, 2>& ev,
                                                            Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const operators::Curl<T, 2> curl{ev, q};
        const auto G = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            // Helmholtz transverse shear γ_α = w_{s,α} + ε_α^β ψ_{,β}: the gradient of the
            // independent shear potential (slot 4) plus the curl of ψ (slot 3).
            const Index idx_ws  = 5 * i + 4;
            B(8*q + 6, idx_ws) = G(i, 0);
            B(8*q + 7, idx_ws) = G(i, 1);

            const Index idx_psi = 5 * i + 3;
            B(8*q + 6, idx_psi) = curl(i, 0);
            B(8*q + 7, idx_psi) = curl(i, 1);
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlinHier5pHelmholtz<T>::constitutive_matrix(const ElementValues<T, 2>& ev,
                                                            Index q) const
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
ShellReissnerMindlinHier5pHelmholtz<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    Matrix<T>& U = this->N_w_;
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    U.setZero(3 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);
            // Physical displacement u = v_b + w_s A_3 (ψ does no work).
            for (Index k = 0; k < 3; ++k)
                U(3 * q + k, 5 * i + k) = Ni;       // v_b translation (slots 0..2)
            for (Index r = 0; r < 3; ++r)
                U(3 * q + r, 5 * i + 4) = Ni * A3(r);  // w_s A_3 (slot 4)
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pHelmholtz<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Director tilt (covariant) w_α = A_α·(Φ_b×A_3) + ε_α^β ψ_{,β} = −v_{b,α}·A_3 + curl ψ.
    Matrix<T>& Nphi = this->N_phi_;
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    Nphi.setZero(2 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        const operators::Curl<T, 2> curl{ev, q};
        const Vector3<T> A3 = ev.normal(q);
        const auto G = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T N_u = G(i, 0), N_v = G(i, 1);
            for (Index k = 0; k < 3; ++k) {
                Nphi(2*q,     5*i + k) = -N_u * A3(k);
                Nphi(2*q + 1, 5*i + k) = -N_v * A3(k);
            }
            Nphi(2*q,     5*i + 3) = curl(i, 0);
            Nphi(2*q + 1, 5*i + 3) = curl(i, 1);
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pHelmholtz<T>::psi(const ElementValues<T, 2>& parent, Matrix<T>& out) const
{
    const Index Q = parent.basis_derivs[0].cols();
    const Index N = parent.basis_derivs[0].rows();
    out.setZero(Q, 5 * N);
    for (Index q = 0; q < Q; ++q) {
        const auto Nf = parent.N(q);
        for (Index i = 0; i < N; ++i)
            out(q, 5 * i + 3) = Nf(i);   // ψ lives in DOF slot 3
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pHelmholtz<T>::psi_gradient(const ElementValues<T, 2>& parent,
                                                     const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    const Index Q = parent.num_points();
    const Index N = static_cast<Index>(parent.basis_derivs[0].rows());
    out.setZero(Q, 5 * N);
    for (Index q = 0; q < Q; ++q) {
        // ∇ψ·dir = ψ_,α (A^α·dir): contravariant raise of dir contracted with the
        // parametric ψ-gradient N^i_,α (ψ in DOF slot 3).
        const Vector3<T> dq = dir.row(q).transpose();
        const auto [d1, d2] = this->contravariant_dir(parent, q, dq);
        const auto grad = parent.dN(q);
        for (Index i = 0; i < N; ++i)
            out(q, 5 * i + 3) = d1 * grad(i, 0) + d2 * grad(i, 1);
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5pHelmholtz<T>::director_variation(const ElementValues<T, 2>& parent,
                                                           const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    // Surface normal a_3 rotates with the bending (Cartesian) displacement only; the ψ
    // and w_s slots are shear potentials, which do not tilt the surface.
    this->surface_director_variation(parent, dir, out);
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlinHier5pHelmholtz<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlinHier5pHelmholtz<float>;
#endif

} // namespace pyck
