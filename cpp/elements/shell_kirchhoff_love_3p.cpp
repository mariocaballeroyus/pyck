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
        throw std::invalid_argument("ShellKirchhoffLove3p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    const ColMatrix<T, 3>& a_3 = ev.n;

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(6 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q){

        // Precomputed geometric primitives
        auto derivs1 = ev.results_[1].col(q);
        const Vec3 A1 = ev.a(0).row(q).transpose(); // reference basis A_1
        const Vec3 A2 = ev.a(1).row(q).transpose(); // reference basis A_2
        const Vec3 A3 = a_3.row(q).transpose();     // unit normal A_3

        const operators::CovariantHessian<T, 2> hess{
            ev.results_, ev.position_data, ev.g_inv_data, q};

        for (Index i = 0; i < N; ++i) {
            // Shape Derivative N^i_{,α}
            const T Nu = derivs1(i * 2 + 0);
            const T Nv = derivs1(i * 2 + 1);
            // Covariant Hessian H_{αβ} = N^i_{,αβ} − Γ^λ_{αβ} N^i_{,λ}
            const T H11 = hess(i, 0, 0);
            const T H22 = hess(i, 1, 1);
            const T H12 = hess(i, 0, 1);

            for (Index k = 0; k < 3; ++k) {
                // Membrane ε_{αβ} = (1/2)(u_{,α}·A_β + u_{,β}·A_α)
                B(6*q,     3*i + k) = Nu * A1(k);
                B(6*q + 1, 3*i + k) = Nv * A2(k);
                B(6*q + 2, 3*i + k) = Nu * A2(k) + Nv * A1(k);

                // Bending κ_{αβ} = −u_{,αβ}·A_3 + Γ^λ_{αβ} (u_{,λ}·A_3)
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
    const Eigen::Matrix<T, 3, 1> g_inv_q = g_inv_voigt(ev, q);

    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(6, 6);
    D.template block<3, 3>(0, 0) = material_->membrane_voigt(g_inv_q);   // t·C
    D.template block<3, 3>(3, 3) = material_->bending_voigt (g_inv_q);   // (t³/12)·C
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Physical displacement u = (u_x, u_y, u_z): the three Cartesian DOFs.
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Matrix<T>& U = this->N_w_workspace_;
    U.setZero(3 * Q, 3 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.results_[0].col(q);
        for (Index i = 0; i < N; ++i) {
            const T Ni = slab0(i);
            U(3 * q + 0, 3 * i + 0) = Ni;   // u_x
            U(3 * q + 1, 3 * i + 1) = Ni;   // u_y
            U(3 * q + 2, 3 * i + 2) = Ni;   // u_z
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
