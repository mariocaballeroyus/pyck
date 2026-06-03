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
ShellKirchhoffLove3p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    // Reset preallocated matrix
    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    B.setZero(6 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q) {
        // Precomputed geometric primitives
        auto derivs1 = ev.basis_derivs[1].col(q);
        const Vector3<T> A1 = ev.a(0).row(q).transpose(); // reference basis A_1
        const Vector3<T> A2 = ev.a(1).row(q).transpose(); // reference basis A_2
        const Vector3<T> A3 = ev.n.row(q).transpose();    // unit normal A_3
        // Discrete operators
        const operators::CovariantHessian<T, 2> hess{ev.basis_derivs, 
                                                     ev.position_derivs, 
                                                     ev.metric_inv, q};

        for (Index i = 0; i < N; ++i) {
            // Shape gradient G_{α} = N^i_{,α}
            const T G1 = derivs1(i * 2 + 0);
            const T G2 = derivs1(i * 2 + 1);
            // Covariant Hessian H_{αβ} = -(1/2)(u_{|αβ}·A_3 + u_{|βα}·A_3)
            const T H11 = hess(i, 0, 0);
            const T H22 = hess(i, 1, 1);
            const T H12 = hess(i, 0, 1);

            for (Index k = 0; k < 3; ++k) {
                // Membrane ε_{αβ} = (1/2)(u_{,α}·A_β + u_{,β}·A_α)
                B(6*q,     3*i + k) = G1 * A1(k);
                B(6*q + 1, 3*i + k) = G2 * A2(k);
                B(6*q + 2, 3*i + k) = G1 * A2(k) + G2 * A1(k);
                // Bending κ_{αβ} = -(1/2)(u_{|αβ}·A_3 + u_{|βα}·A_3)
                B(6*q + 3, 3*i + k) = -H11 * A3(k);
                B(6*q + 4, 3*i + k) = -H22 * A3(k);
                B(6*q + 5, 3*i + k) = -T(2) * H12 * A3(k);
            }
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellKirchhoffLove3p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    const StaticVector<T, 3> metric_inv_q = ev.metric_inv_voigt(q);

    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(6, 6);
    D.template block<3, 3>(0, 0) = material_->membrane_voigt(metric_inv_q);   // t·C
    D.template block<3, 3>(3, 3) = material_->bending_voigt (metric_inv_q);   // (t³/12)·C
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Reset preallocated matrix
    const Index Q = ev.basis_derivs[0].cols();
    const Index N = ev.basis_derivs[0].rows();
    Matrix<T>& Nw = this->N_w_workspace_;
    Nw.setZero(3 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < N; ++i) {
            for (Index k = 0; k < 3; ++k) {
                // Cartesian displacement u
                Nw(3 * q + k, 3 * i + k) = ev.basis_derivs[0].col(q)(i);
            }
        }
    }
}

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::rotation_shape_matrix(const ElementValues<T, 2>&) const
{
    throw std::runtime_error("ShellKirchhoffLove3p::rotation_shape_matrix: "
                             "not implemented.");
}

// === Template Instantiations ========================================================

template class ShellKirchhoffLove3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellKirchhoffLove3p<float>;
#endif

} // namespace pyck
