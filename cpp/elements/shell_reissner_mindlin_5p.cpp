#include "shell_reissner_mindlin_5p.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_gradient.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlin5p<T>::ShellReissnerMindlin5p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlin5p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::membrane_strain_matrix(const ElementValues<T, 2>& ev,
                                                  Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const auto cov = ev.cov_basis(q);
        const Vector3<T> A1 = cov(0), A2 = cov(1);
        const auto G = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T G1i = G(i, 0), G2i = G(i, 1);
            // Membrane ε_{αβ} = (1/2)(u_{,α}·A_β + u_{,β}·A_α)
            for (Index k = 0; k < 3; ++k) {
                B(8*q,     5*i + k) = G1i * A1(k);
                B(8*q + 1, 5*i + k) = G2i * A2(k);
                B(8*q + 2, 5*i + k) = G1i * A2(k) + G2i * A1(k);
            }
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::bending_strain_matrix(const ElementValues<T, 2>& ev,
                                                 Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        // Unit normal 1st derivative A_{3,β}
        const auto nd = ev.normal_deriv(q);
        const Vector3<T> A3_d1 = nd(0), A3_d2 = nd(1);
        const auto G = ev.dN(q);
        // Covariant vector gradient φ_{α|β}
        const operators::CovariantGradient<T, 2> vgrad{ev, q};

        for (Index i = 0; i < N; ++i) {
            const T G1i = G(i, 0), G2i = G(i, 1);
            const T D111 = vgrad(i, 0, 0, 0), D112 = vgrad(i, 0, 0, 1),
                    D121 = vgrad(i, 0, 1, 0), D122 = vgrad(i, 0, 1, 1),
                    D211 = vgrad(i, 1, 0, 0), D212 = vgrad(i, 1, 0, 1),
                    D221 = vgrad(i, 1, 1, 0), D222 = vgrad(i, 1, 1, 1);

            // Bending κ_{αβ} += (1/2)(u_{,α}·A_{3,β} + u_{,β}·A_{3,α}) — Cartesian u.
            for (Index k = 0; k < 3; ++k) {
                B(8*q + 3, 5*i + k) = G1i * A3_d1(k);
                B(8*q + 4, 5*i + k) = G2i * A3_d2(k);
                B(8*q + 5, 5*i + k) = G1i * A3_d2(k) + G2i * A3_d1(k);
            }
            // Bending κ_{αβ} += (1/2)(φ_{α|β} + φ_{β|α}) — contravariant rotations.
            B(8*q + 3, 5*i + 3) = D111;
            B(8*q + 3, 5*i + 4) = D211;
            B(8*q + 4, 5*i + 3) = D122;
            B(8*q + 4, 5*i + 4) = D222;
            B(8*q + 5, 5*i + 3) = D112 + D121;
            B(8*q + 5, 5*i + 4) = D212 + D221;
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::shear_strain_matrix(const ElementValues<T, 2>& ev,
                                               Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        const auto A = ev.metric(q);
        const T A11 = A(0, 0), A22 = A(1, 1), A12 = A(0, 1);
        const auto Nf = ev.N(q);
        const auto G  = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni  = Nf(i);
            const T G1i = G(i, 0), G2i = G(i, 1);
            // Transverse shear 2ε_{α3} += u_{,α}·A_3 — Cartesian u.
            for (Index k = 0; k < 3; ++k) {
                B(8*q + 6, 5*i + k) = G1i * A3(k);
                B(8*q + 7, 5*i + k) = G2i * A3(k);
            }
            // Transverse shear 2ε_{α3} += φ_α = φ^λ A_{λα} — contravariant rotations.
            B(8*q + 6, 5*i + 3) = Ni * A11;
            B(8*q + 6, 5*i + 4) = Ni * A12;
            B(8*q + 7, 5*i + 3) = Ni * A12;
            B(8*q + 7, 5*i + 4) = Ni * A22;
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlin5p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
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
ShellReissnerMindlin5p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset shape matrix values
    Matrix<T>& N_w = this->N_w_;
    N_w.setZero(3 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        // Shape N^i
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);

            // Cartesian displacement u
            N_w(3 * q + 0, 5 * i + 0) = Ni;
            N_w(3 * q + 1, 5 * i + 1) = Ni;
            N_w(3 * q + 2, 5 * i + 2) = Ni;
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    // Reset shape matrix values
    Matrix<T>& N_phi = this->N_phi_;
    N_phi.setZero(2 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q) {
        // Covariant metric
        const auto A = ev.metric(q);
        const T A11 = A(0, 0), A22 = A(1, 1), A12 = A(0, 1);

        // Shape N^i
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);

            // θ_1 = φ^1 A_{11} + φ^2 A_{21}
            N_phi(2*q,     5*i + 3) = Ni * A11;
            N_phi(2*q,     5*i + 4) = Ni * A12;
            // θ_2 = φ^1 A_{12} + φ^2 A_{22}
            N_phi(2*q + 1, 5*i + 3) = Ni * A12;
            N_phi(2*q + 1, 5*i + 4) = Ni * A22;
        }
    }
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin5p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin5p<float>;
#endif

} // namespace pyck
