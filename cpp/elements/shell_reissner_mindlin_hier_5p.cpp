#include "shell_reissner_mindlin_hier_5p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"
#include "../operators/covariant_gradient.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlinHier5p<T>::ShellReissnerMindlinHier5p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlinHier5p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellReissnerMindlinHier5p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset strain matrix values
    Matrix<T>& B_voigt = this->B_voigt_;
    B_voigt.setZero(8 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        // Covariant basis, unit normal, and midsurface metric
        const auto cov = ev.cov_basis(q);
        const Vector3<T> A1 = cov(0), A2 = cov(1), A3 = ev.normal(q);
        const auto A = ev.metric(q);
        const T A11 = A(0, 0), A22 = A(1, 1), A12 = A(0, 1);

        // Shape N^i, shape gradient N^i_{,α}
        const auto Nf = ev.N(q);
        const auto G  = ev.dN(q);
        // Covariant Hessian H^i_{αβ} (Kirchhoff-Love bending of v)
        const operators::CovariantHessian<T, 2> hess{ev, q};
        // Covariant gradient w_{α|β} (bending contribution of w)
        const operators::CovariantGradient<T, 2> wgrad{ev, q};

        for (Index i = 0; i < N; ++i) {
            const T Ni  = Nf(i);
            const T G1i = G(i, 0), G2i = G(i, 1);
            const T H11 = hess(i, 0, 0), H22 = hess(i, 1, 1), H12 = hess(i, 0, 1);

            // --- Cartesian displacement v (Kirchhoff-Love part, no shear) -----------

            for (Index k = 0; k < 3; ++k) {
                // Membrane ε_{αβ} = (1/2)(v_{,α}·A_β + v_{,β}·A_α)
                B_voigt(8*q,     5*i + k) = G1i * A1(k);
                B_voigt(8*q + 1, 5*i + k) = G2i * A2(k);
                B_voigt(8*q + 2, 5*i + k) = G1i * A2(k) + G2i * A1(k);
                // Bending κ_{αβ} = -(1/2)(v_{|αβ} + v_{|βα})·A_3
                B_voigt(8*q + 3, 5*i + k) =        -H11 * A3(k);
                B_voigt(8*q + 4, 5*i + k) =        -H22 * A3(k);
                B_voigt(8*q + 5, 5*i + k) = -T(2) * H12 * A3(k);
                // Transverse shear vanishes for the Kirchhoff-Love field
            }

            // --- Hierarchic difference vector w = w^λ A_λ (λ = slots 3, 4) ----------

            for (Index lam = 0; lam < 2; ++lam) {
                const Index idx = 5 * i + 3 + lam;
                // Bending κ_{αβ} += (1/2)(w_{α|β} + w_{β|α})
                // ε_{αβ} = ε^3p_{αβ} + θ³ w_{,α}·A_β.
                B_voigt(8*q + 3, idx) = wgrad(i, lam, 0, 0);
                B_voigt(8*q + 4, idx) = wgrad(i, lam, 1, 1);
                B_voigt(8*q + 5, idx) = wgrad(i, lam, 0, 1) + wgrad(i, lam, 1, 0);
            }
            // Transverse shear 2ε_{α3} = w·A_α = N^i w^λ A_{λα}
            B_voigt(8*q + 6, 5*i + 3) = Ni * A11;
            B_voigt(8*q + 6, 5*i + 4) = Ni * A12;
            B_voigt(8*q + 7, 5*i + 3) = Ni * A12;
            B_voigt(8*q + 7, 5*i + 4) = Ni * A22;
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlinHier5p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
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
ShellReissnerMindlinHier5p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset shape matrix values
    Matrix<T>& N_w = this->N_w_;
    N_w.setZero(3 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        const auto Nf = ev.N(q);
        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);
            // Mid-surface displacement u = v (the Cartesian part)
            N_w(3 * q + 0, 5 * i + 0) = Ni;
            N_w(3 * q + 1, 5 * i + 1) = Ni;
            N_w(3 * q + 2, 5 * i + 2) = Ni;
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlinHier5p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset shape matrix values
    Matrix<T>& N_phi = this->N_phi_;
    N_phi.setZero(2 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto A = ev.metric(q);
        const T A11 = A(0, 0), A22 = A(1, 1), A12 = A(0, 1);
        const auto Nf = ev.N(q);
        const auto G  = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni  = Nf(i);
            const T G1i = G(i, 0), G2i = G(i, 1);

            // Kirchhoff-Love rotation θ_α = (Φ×A₃)·A_α = −v_{,α}·A_3
            for (Index k = 0; k < 3; ++k) {
                N_phi(2*q,     5*i + k) = -G1i * A3(k);
                N_phi(2*q + 1, 5*i + k) = -G2i * A3(k);
            }
            // Difference-vector tilt θ_α += w·A_α = N^i w^λ A_{λα}
            N_phi(2*q,     5*i + 3) = Ni * A11;
            N_phi(2*q,     5*i + 4) = Ni * A12;
            N_phi(2*q + 1, 5*i + 3) = Ni * A12;
            N_phi(2*q + 1, 5*i + 4) = Ni * A22;
        }
    }
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlinHier5p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlinHier5p<float>;
#endif

} // namespace pyck
