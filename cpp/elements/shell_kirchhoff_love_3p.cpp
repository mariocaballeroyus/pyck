#include "shell_kirchhoff_love_3p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "covariant_hessian.hpp"

namespace pyck
{

template <std::floating_point T>
ShellKirchhoffLove3p<T>::ShellKirchhoffLove3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellKirchhoffLove3p: "
                                    "material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::membrane_strain_matrix(const ElementValues<T, 2>& ev,
                                                Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        // Covariant basis
        const auto A = ev.cov_basis(q);
        const Vector3<T> A1 = A(0), A2 = A(1);
        const auto grad = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T G1i = grad(i, 0), G2i = grad(i, 1);
            // Membrane ε_{αβ} = (1/2)(u_{,α}·A_β + u_{,β}·A_α)
            for (Index k = 0; k < 3; ++k) {
                B(6*q,     3*i + k) = G1i * A1(k);
                B(6*q + 1, 3*i + k) = G2i * A2(k);
                B(6*q + 2, 3*i + k) = G1i * A2(k) + G2i * A1(k);
            }
        }
    }
}

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::bending_strain_matrix(const ElementValues<T, 2>& ev,
                                               Matrix<T>& B) const
{
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();

    for (Index q = 0; q < Q; ++q) {
        const Vector3<T> A3 = ev.normal(q);
        // Covariant Hessian H^i_{αβ}
        const operators::CovariantHessian<T, 2> hess{ev, q};

        for (Index i = 0; i < N; ++i) {
            const T H11i = hess(i, 0, 0), H22i = hess(i, 1, 1), H12i = hess(i, 0, 1);
            // Bending κ_{αβ} = -(1/2)(u_{|αβ}·A_3 + u_{|βα}·A_3)
            for (Index k = 0; k < 3; ++k) {
                B(6*q + 3, 3*i + k) = -H11i * A3(k);
                B(6*q + 4, 3*i + k) = -H22i * A3(k);
                B(6*q + 5, 3*i + k) = -T(2) * H12i * A3(k);
            }
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellKirchhoffLove3p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    // Transform elasticity matrix into curvilinear coordinate system
    const StaticVector<T, 3> metric_inv = ev.metric_inv_voigt(q);
    const Eigen::Matrix<T, 3, 3> C = material_->elasticity_voigt(metric_inv);

    // Scale material matrices (no shear for KL)
    const T t = material_->thickness();
    const Eigen::Matrix<T, 3, 3> Dm = t * C;
    const Eigen::Matrix<T, 3, 3> Db = (t * t * t / T(12)) * C;

    // Assemble into full consitutive matrix
    ConstitutiveMatrix<T> D_voigt = ConstitutiveMatrix<T>::Zero(6, 6);
    D_voigt.template block<3, 3>(0, 0) = Dm;
    D_voigt.template block<3, 3>(3, 3) = Db;
    return D_voigt;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of basis and points
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset strain matrix values
    Matrix<T>& N_w = this->N_w_;
    N_w.setZero(3 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q) {
        // Shape values N^i
        const auto Nf = ev.N(q);

        for (Index i = 0; i < N; ++i) {
            const T Ni = Nf(i);
            for (Index k = 0; k < 3; ++k) {
                // Cartesian displacement u
                N_w(3 * q + k, 3 * i + k) = Ni;
            }
        }
    }
}

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Number of points and basis
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    // Reset shape matrix values
    Matrix<T>& N_phi = this->N_phi_;
    N_phi.setZero(2 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q) {
        // Unit normal
        const Vector3<T> A3 = ev.normal(q);

        // Shape gradient N^i_{,α} 
        const auto grad = ev.dN(q);

        for (Index i = 0; i < N; ++i) {
            const T G1i = grad(i, 0), G2i = grad(i, 1);

            for (Index k = 0; k < 3; ++k) {
                // θ_α = −N^i_{,α} (A_3)_k u_k
                N_phi(2*q,     3*i + k) = -G1i * A3(k);
                N_phi(2*q + 1, 3*i + k) = -G2i * A3(k);
            }
        }
    }
}

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::director_variation(const ElementValues<T, 2>& parent,
                                            const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    // a_3 is the surface normal; its variation is the covariant tilt θ_α = −u_,α·A_3.
    // For this rotation-free shell that is the whole `rotation` trace.
    this->surface_director_variation(parent, dir, out);
}

// === Template Instantiations ========================================================

template class ShellKirchhoffLove3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellKirchhoffLove3p<float>;
#endif

} // namespace pyck
