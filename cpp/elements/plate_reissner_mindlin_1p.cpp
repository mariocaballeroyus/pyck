#include "plate_reissner_mindlin_1p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"
#include "../operators/laplace_beltrami.hpp"
#include "../operators/laplace_beltrami_gradient.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlin1p<T>::PlateReissnerMindlin1p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlin1p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
ConstitutiveMatrix<T>
PlateReissnerMindlin1p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    // D = [ D_b   0   ]
    //     [ 0     D_s ]
    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(5, 5);
    D.template topLeftCorner    <3, 3>() = material_->bending_voigt(g_inv_voigt(ev, q));
    D.template bottomRightCorner<2, 2>() = material_->shear_voigt  (g_inv_voigt(ev, q));
    return D;
}

template <std::floating_point T>
void
PlateReissnerMindlin1p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(5 * Q, N);

    operators::CovariantHessian<T, 2>         hess {ev.results_, ev.position_data, ev.g_inv_data};
    operators::LaplaceBeltramiGradient<T, 2> lgrad{ev.results_, ev.lb_grad_conn_, ev.g_inv_data};

    for (Index q = 0; q < Q; ++q)
        for (Index i = 0; i < N; ++i)
        {
            // Bending = −(covariant Hessian of w): [−H_11; −H_22; −2H_12].
            B(5*q,     i) = -hess(i, 0, 0, q);
            B(5*q + 1, i) = -hess(i, 1, 1, q);
            B(5*q + 2, i) = -T(2) * hess(i, 0, 1, q);

            // Shear = −(K_b/K_s)(Δ_g N_i)_{,α} = −ratio · P_{iα}.
            B(5*q + 3, i) = -ratio * lgrad(i, 0, q);
            B(5*q + 4, i) = -ratio * lgrad(i, 1, q);
        }
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
PlateReissnerMindlin1p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& Nw = this->N_w_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Nw.setZero(3 * Q, N);   // u = (0, 0, w): transverse deflection along +z

    operators::LaplaceBeltrami<T, 2> lapb{ev.results_, ev.position_data, ev.g_inv_data};

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = ev.results_[0].col(q);
        for (Index i = 0; i < N; ++i)
            // w = N_i − (K_b/K_s) Δ_g N_i = N_i − ratio·L_i, in the z-component row.
            Nw(3 * q + 2, i) = slab0(i) - ratio * lapb(i, q);
    }
}

template <std::floating_point T>
void
PlateReissnerMindlin1p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // N_rot = [ -N_{i|1} ]
    //         [ -N_{i|2} ]
    Matrix<T>& Nphi = this->N_phi_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Nphi.resize(2 * Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = ev.results_[1].col(q);
        for (Index i = 0; i < N; ++i) {
            Nphi(2*q,     i) = -slab1(i * 2 + 0);
            Nphi(2*q + 1, i) = -slab1(i * 2 + 1);
        }
    }
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin1p<float>;
#endif

} // namespace pyck
