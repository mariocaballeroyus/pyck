#include "plate_kirchhoff_love_1p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"

namespace pyck
{

template <std::floating_point T>
PlateKirchhoffLove1p<T>::PlateKirchhoffLove1p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateKirchhoffLove1p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
PlateKirchhoffLove1p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(3 * Q, N);

    operators::CovariantHessian<T, 2> hess{ev.results_, ev.position_data, ev.g_inv_data};

    for (Index q = 0; q < Q; ++q)
        for (Index i = 0; i < N; ++i)
        {
            // Bending strain = −(covariant Hessian of w): B_i = [−H_11; −H_22; −2H_12].
            B(3*q,     i) = -hess(i, 0, 0, q);
            B(3*q + 1, i) = -hess(i, 1, 1, q);
            B(3*q + 2, i) = -T(2) * hess(i, 0, 1, q);
        }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
PlateKirchhoffLove1p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    // D = D_b. No shear block (normality assumption).
    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(3, 3);
    D.template topLeftCorner<3, 3>() = material_->bending_voigt(g_inv_voigt(ev, q));
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
PlateKirchhoffLove1p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Physical displacement u = (0, 0, w): transverse deflection along +z.
    Matrix<T>& N_w = this->N_w_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    N_w.setZero(3 * Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.results_[0].col(q);
        for (Index i = 0; i < N; ++i) {
            N_w(3 * q + 2, i) = slab0(i);
        }
    }
}

template <std::floating_point T>
void
PlateKirchhoffLove1p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // N_rot = [ -N_{i|1} ]
    //         [ -N_{i|2} ]
    Matrix<T>& N_varphi = this->N_phi_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    N_varphi.resize(2 * Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = ev.results_[1].col(q);
        for (Index i = 0; i < N; ++i) {
            N_varphi(2*q,     i) = -slab1(i * 2 + 0);
            N_varphi(2*q + 1, i) = -slab1(i * 2 + 1);
        }
    }
}

// === Template Instantiations ========================================================

template class PlateKirchhoffLove1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateKirchhoffLove1p<float>;
#endif

} // namespace pyck
