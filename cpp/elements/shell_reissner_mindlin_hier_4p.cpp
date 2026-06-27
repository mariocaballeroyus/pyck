#include "shell_reissner_mindlin_hier_4p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"
#include "../operators/laplace_beltrami.hpp"
#include "../operators/laplace_beltrami_gradient.hpp"
#include "../operators/curl.hpp"
#include "../operators/curl_gradient.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlinHier4p<T>::ShellReissnerMindlinHier4p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlinHier4p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellReissnerMindlinHier4p<T>::membrane_strain_matrix(const ElementValues<T, 2>& ev,
                                                      Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    for (Index q = 0; q < Q; ++q) {
        const auto Bc = ev.curvature(q);
        const T B11 = Bc(0, 0), B22 = Bc(1, 1), B12 = Bc(0, 1);
        const auto A = ev.cov_basis(q);
        const Vector3<T> A1 = A(0), A2 = A(1), A3 = ev.normal(q);
        const auto G = ev.dN(q);
        const operators::LaplaceBeltrami<T, 2> lapb{ev, q};

        for (Index i = 0; i < N; ++i) {
            const T N_u = G(i, 0), N_v = G(i, 1);
            const T L = lapb(i);
            // Membrane ε_{αβ}: Kirchhoff-Love A_α·v_{,β}; shear enrichment += -B_{αβ} w_s.
            for (Index k = 0; k < 3; ++k) {
                const Index idx = 4 * i + k;
                const T ws = -ratio * L * A3(k);
                B(8*q + 0, idx) = A1(k) * N_u                - B11 * ws;
                B(8*q + 1, idx) = A2(k) * N_v                - B22 * ws;
                B(8*q + 2, idx) = A1(k) * N_v + A2(k) * N_u  - T(2) * B12 * ws;
            }
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier4p<T>::bending_strain_matrix(const ElementValues<T, 2>& ev,
                                                     Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto nd = ev.normal_deriv(q);
        const Vector3<T> A3_d1 = nd(0), A3_d2 = nd(1);
        // Third fundamental form (B²)_{αβ} = A_{3,α}·A_{3,β}
        const T B2_11 = A3_d1.dot(A3_d1), B2_22 = A3_d2.dot(A3_d2), B2_12 = A3_d1.dot(A3_d2);
        const operators::CovariantHessian<T, 2> hess{ev, q};
        const operators::LaplaceBeltrami<T, 2>  lapb{ev, q};
        const operators::CurlGradient<T, 2>     curlgrad{ev, q};

        for (Index i = 0; i < N; ++i) {
            const T H11 = hess(i, 0, 0), H22 = hess(i, 1, 1), H12 = hess(i, 0, 1);
            const T L = lapb(i);
            // Bending κ_{αβ}: Kirchhoff-Love −v_{|αβ}·A_3; shear enrichment += w_s (B²)_{αβ}.
            for (Index k = 0; k < 3; ++k) {
                const Index idx = 4 * i + k;
                const T ws = -ratio * L * A3(k);
                B(8*q + 3, idx) =        -H11 * A3(k)       + ws * B2_11;
                B(8*q + 4, idx) =        -H22 * A3(k)       + ws * B2_22;
                B(8*q + 5, idx) = -T(2) * H12 * A3(k) + T(2) * ws * B2_12;
            }
            // Twist potential ψ: κ_{αβ} += (curl ψ)_{(α|β)}
            const Index idx_psi = 4 * i + 3;
            B(8*q + 3, idx_psi) = curlgrad(i, 0, 0);
            B(8*q + 4, idx_psi) = curlgrad(i, 1, 1);
            B(8*q + 5, idx_psi) = curlgrad(i, 0, 1) + curlgrad(i, 1, 0);
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier4p<T>::shear_strain_matrix(const ElementValues<T, 2>& ev,
                                                   Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto nd = ev.normal_deriv(q);
        const Vector3<T> A3_d1 = nd(0), A3_d2 = nd(1);
        const operators::Curl<T, 2>                    curl {ev, q};
        const operators::LaplaceBeltrami<T, 2>         lapb{ev, q};
        const operators::LaplaceBeltramiGradient<T, 2> lgrad{ev, q};

        for (Index i = 0; i < N; ++i) {
            // w_s of DOF (i,k) is -(K_b/K_s)(Δ N_i) A_3(k); the transverse shear is its surface
            // gradient γ_α = ∂_α(w_s) = -(K_b/K_s)[ (Δ N_i)_{,α} A_3(k) + (Δ N_i) A_{3,α}(k) ].
            // The second (Weingarten) product term is O(curvature), vanishing on a flat plate.
            const T L  = lapb(i);
            const T P1 = lgrad(i, 0);
            const T P2 = lgrad(i, 1);
            for (Index k = 0; k < 3; ++k) {
                const Index idx = 4 * i + k;
                B(8*q + 6, idx) = -ratio * (P1 * A3(k) + L * A3_d1(k));
                B(8*q + 7, idx) = -ratio * (P2 * A3(k) + L * A3_d2(k));
            }
            // Twist potential ψ: γ_α += ε_α^β ψ_{,β}
            const Index idx_psi = 4 * i + 3;
            B(8*q + 6, idx_psi) = curl(i, 0);
            B(8*q + 7, idx_psi) = curl(i, 1);
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlinHier4p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
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
ShellReissnerMindlinHier4p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& U = this->N_w_;
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    U.setZero(3 * Q, 4 * N);

    for (Index q = 0; q < Q; ++q) {
        const operators::LaplaceBeltrami<T, 2> lapb{ev, q};
        const Vector3<T> A3 = ev.normal(q);
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);
            const T L  = lapb(i);

            // Physical displacement u = v_b + w_s A_3 (ψ does no work).
            for (Index k = 0; k < 3; ++k) {
                const T ws = -ratio * L * A3(k);
                for (Index r = 0; r < 3; ++r) {
                    U(3 * q + r, 4 * i + k) = (r == k ? Ni : T(0)) + ws * A3(r);
                }
            }
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier4p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Director tilt (covariant) w_α = A_α·(Φ_b×A_3) + ε_α^β ψ_{,β} = −v_{b,α}·A_3 + curl ψ.
    Matrix<T>& Nphi = this->N_phi_;
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    Nphi.setZero(2 * Q, 4 * N);

    for (Index q = 0; q < Q; ++q) {
        const operators::Curl<T, 2> curl{ev, q};
        const Vector3<T> A3 = ev.normal(q);
        const auto G = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T N_u = G(i, 0), N_v = G(i, 1);
            for (Index k = 0; k < 3; ++k) {
                Nphi(2*q,     4*i + k) = -N_u * A3(k);
                Nphi(2*q + 1, 4*i + k) = -N_v * A3(k);
            }
            Nphi(2*q,     4*i + 3) = curl(i, 0);
            Nphi(2*q + 1, 4*i + 3) = curl(i, 1);
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier4p<T>::psi(const ElementValues<T, 2>& parent, Matrix<T>& out) const
{
    const Index Q = parent.basis_derivs[0].cols();
    const Index N = parent.basis_derivs[0].rows();
    out.setZero(Q, 4 * N);
    for (Index q = 0; q < Q; ++q) {
        const auto Nf = parent.N(q);
        for (Index i = 0; i < N; ++i)
            out(q, 4 * i + 3) = Nf(i);   // ψ lives in DOF slot 3
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier4p<T>::psi_gradient(const ElementValues<T, 2>& parent,
                                            const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    const Index Q = parent.num_points();
    const Index N = static_cast<Index>(parent.basis_derivs[0].rows());
    out.setZero(Q, 4 * N);
    for (Index q = 0; q < Q; ++q) {
        // ∇ψ·dir = ψ_,α (A^α·dir): contravariant raise of dir contracted with the
        // parametric ψ-gradient N^i_,α (ψ in DOF slot 3).
        const Vector3<T> dq = dir.row(q).transpose();
        const auto [d1, d2] = this->contravariant_dir(parent, q, dq);
        const auto grad = parent.dN(q);
        for (Index i = 0; i < N; ++i)
            out(q, 4 * i + 3) = d1 * grad(i, 0) + d2 * grad(i, 1);
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier4p<T>::director_variation(const ElementValues<T, 2>& parent,
                                                  const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    // Surface normal a_3 rotates with the bending (Cartesian) displacement only; the ψ
    // slot is the shear potential, which does not tilt the surface.
    this->surface_director_variation(parent, dir, out);
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlinHier4p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlinHier4p<float>;
#endif

} // namespace pyck
