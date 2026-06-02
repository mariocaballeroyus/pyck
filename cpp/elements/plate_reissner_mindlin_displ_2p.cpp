#include "plate_reissner_mindlin_displ_2p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"
#include "../operators/covariant_hessian.hpp"
#include "../operators/laplace_beltrami.hpp"
#include "../operators/laplace_beltrami_gradient.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlinDispl2p<T>::PlateReissnerMindlinDispl2p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlinDispl2p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
PlateReissnerMindlinDispl2p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(5 * Q, 2 * N);

    operators::CovariantHessian<T, 2>         hess {ev.results_, ev.position_data, ev.g_inv_data};
    operators::LaplaceBeltramiGradient<T, 2> lgrad{ev.results_, ev.lb_grad_conn_, ev.g_inv_data};

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = ev.results_[1].col(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_u_i = slab1(i * 2 + 0);
            const T N_v_i = slab1(i * 2 + 1);

            // B_b = [ -N_{i|11}    N_{i|12}          ]   κ_{11}
            //       [ -N_{i|22}   -N_{i|12}          ]   κ_{22}
            //       [ -2 N_{i|12}  N_{i|22}−N_{i|11} ]   2κ_{12}
            B(5*q,     2*i    ) = -hess(i, 0, 0, q);
            B(5*q,     2*i + 1) =  hess(i, 0, 1, q);
            B(5*q + 1, 2*i    ) = -hess(i, 1, 1, q);
            B(5*q + 1, 2*i + 1) = -hess(i, 0, 1, q);
            B(5*q + 2, 2*i    ) = -T(2) * hess(i, 0, 1, q);
            B(5*q + 2, 2*i + 1) =  hess(i, 1, 1, q) - hess(i, 0, 0, q);

            // B_s = [ -(K_b/K_s) (Δ_g N_i)_{|1}   N_{i|2} ]   γ_1
            //       [ -(K_b/K_s) (Δ_g N_i)_{|2}  -N_{i|1} ]   γ_2
            B(5*q + 3, 2*i    ) = -ratio * lgrad(i, 0, q);
            B(5*q + 3, 2*i + 1) =  N_v_i;
            B(5*q + 4, 2*i    ) = -ratio * lgrad(i, 1, q);
            B(5*q + 4, 2*i + 1) = -N_u_i;
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
PlateReissnerMindlinDispl2p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(5, 5);
    D.template topLeftCorner    <3, 3>() = material_->bending_voigt(g_inv_voigt(ev, q));
    D.template bottomRightCorner<2, 2>() = material_->shear_voigt  (g_inv_voigt(ev, q));
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
PlateReissnerMindlinDispl2p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& Nw = this->N_w_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Nw.setZero(3 * Q, 2 * N);   // u = (0, 0, w): transverse deflection along +z

    operators::LaplaceBeltrami<T, 2> lapb{ev.results_, ev.position_data, ev.g_inv_data};

    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.results_[0].col(q);
        for (Index i = 0; i < N; ++i)
            // w = N_i − (K_b/K_s) Δ_g N_i = N_i − ratio·L_i, in the z-component row.
            Nw(3 * q + 2, 2*i) = slab0(i) - ratio * lapb(i, q);
    }
}

template <std::floating_point T>
void
PlateReissnerMindlinDispl2p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // N_rot = [ -N_{i|1}   N_{i|2} ]
    //         [ -N_{i|2}  -N_{i|1} ]
    Matrix<T>& Nphi = this->N_phi_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Nphi.setZero(2 * Q, 2 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = ev.results_[1].col(q);
        for (Index i = 0; i < N; ++i) {
            const T N_u_i = slab1(i * 2 + 0);
            const T N_v_i = slab1(i * 2 + 1);
            Nphi(2*q,     2*i    ) = -N_u_i;
            Nphi(2*q,     2*i + 1) =  N_v_i;
            Nphi(2*q + 1, 2*i    ) = -N_v_i;
            Nphi(2*q + 1, 2*i + 1) = -N_u_i;
        }
    }
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlinDispl2p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlinDispl2p<float>;
#endif

} // namespace pyck
