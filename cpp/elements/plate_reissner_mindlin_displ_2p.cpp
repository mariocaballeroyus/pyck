#include "plate_reissner_mindlin_displ_2p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"
#include "laplace_beltrami.hpp"

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
    auto aux = compute_laplace_grad_aux(ev);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(5 * Q, 2 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = ev.results_[1].col(q);
        auto slab2 = ev.results_[2].col(q);
        auto slab3 = ev.results_[3].col(q);

        const T Gam1_11 = ev.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ev.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ev.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ev.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ev.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ev.Gamma(1, 1, 1)(q);

        const T G11 = ev.g_inv(0, 0)(q);
        const T G12 = ev.g_inv(0, 1)(q);
        const T G22 = ev.g_inv(1, 1)(q);
        const T G11_d1 = aux.G_inv_d[0][0][0](q);
        const T G12_d1 = aux.G_inv_d[0][1][0](q);
        const T G22_d1 = aux.G_inv_d[1][1][0](q);
        const T G11_d2 = aux.G_inv_d[0][0][1](q);
        const T G12_d2 = aux.G_inv_d[0][1][1](q);
        const T G22_d2 = aux.G_inv_d[1][1][1](q);
        const T c1 = aux.c[0](q),    c2 = aux.c[1](q);
        const T c1_d1 = aux.c_d[0][0](q), c2_d1 = aux.c_d[1][0](q);
        const T c1_d2 = aux.c_d[0][1](q), c2_d2 = aux.c_d[1][1](q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_u_i    = slab1(i * 2 + 0);
            const T N_v_i    = slab1(i * 2 + 1);
            const T N_uu_i   = slab2(i * 3 + 0);
            const T N_vv_i   = slab2(i * 3 + 1);
            const T N_uv_i   = slab2(i * 3 + 2);
            const T N_uuu_i  = slab3(i * 4 + 0);
            const T N_uuv_i  = slab3(i * 4 + 1);
            const T N_uvv_i  = slab3(i * 4 + 2);
            const T N_vvv_i  = slab3(i * 4 + 3);

            const T N11 = N_uu_i - Gam1_11 * N_u_i - Gam2_11 * N_v_i;
            const T N12 = N_uv_i - Gam1_12 * N_u_i - Gam2_12 * N_v_i;
            const T N22 = N_vv_i - Gam1_22 * N_u_i - Gam2_22 * N_v_i;

            const T lap_1 = ratio * (
                  G11_d1 * N_uu_i  + T(2)*G12_d1 * N_uv_i + G22_d1 * N_vv_i
                + G11    * N_uuu_i + T(2)*G12    * N_uuv_i + G22    * N_uvv_i
                - c1_d1  * N_u_i   - c2_d1       * N_v_i
                - c1     * N_uu_i  - c2          * N_uv_i);
            const T lap_2 = ratio * (
                  G11_d2 * N_uu_i  + T(2)*G12_d2 * N_uv_i + G22_d2 * N_vv_i
                + G11    * N_uuv_i + T(2)*G12    * N_uvv_i + G22    * N_vvv_i
                - c1_d2  * N_u_i   - c2_d2       * N_v_i
                - c1     * N_uv_i  - c2          * N_vv_i);

            // B_b = [ -N_{i|11}    N_{i|12}          ]   κ_{11}
            //       [ -N_{i|22}   -N_{i|12}          ]   κ_{22}
            //       [ -2 N_{i|12}  N_{i|22}−N_{i|11} ]   2κ_{12}
            B(5*q,     2*i    ) = -N11;
            B(5*q,     2*i + 1) =  N12;
            B(5*q + 1, 2*i    ) = -N22;
            B(5*q + 1, 2*i + 1) = -N12;
            B(5*q + 2, 2*i    ) = -T(2) * N12;
            B(5*q + 2, 2*i + 1) =  N22 - N11;

            // B_s = [ -(K_b/K_s) (Δ_g N_i)_{|1}   N_{i|2} ]   γ_1
            //       [ -(K_b/K_s) (Δ_g N_i)_{|2}  -N_{i|1} ]   γ_2
            B(5*q + 3, 2*i    ) = -lap_1;
            B(5*q + 3, 2*i + 1) =  N_v_i;
            B(5*q + 4, 2*i    ) = -lap_2;
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
    Nw.setZero(Q, 2 * N);

    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.results_[0].col(q);
        auto slab1 = ev.results_[1].col(q);
        auto slab2 = ev.results_[2].col(q);

        const T gi11 = ev.g_inv(0, 0)(q);
        const T gi12 = ev.g_inv(0, 1)(q);
        const T gi22 = ev.g_inv(1, 1)(q);
        const T Gam1_11 = ev.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ev.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ev.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ev.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ev.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ev.Gamma(1, 1, 1)(q);

        for (Index i = 0; i < N; ++i) {
            const T N_i    = slab0(i);
            const T N_u_i  = slab1(i * 2 + 0);
            const T N_v_i  = slab1(i * 2 + 1);
            const T N_uu_i = slab2(i * 3 + 0);
            const T N_vv_i = slab2(i * 3 + 1);
            const T N_uv_i = slab2(i * 3 + 2);

            const T N11 = N_uu_i - Gam1_11 * N_u_i - Gam2_11 * N_v_i;
            const T N12 = N_uv_i - Gam1_12 * N_u_i - Gam2_12 * N_v_i;
            const T N22 = N_vv_i - Gam1_22 * N_u_i - Gam2_22 * N_v_i;

            // N_w = [ N_i − (K_b/K_s) Δ_g N_i   0 ]
            Nw(q, 2*i) = N_i - ratio * (gi11 * N11 + T(2) * gi12 * N12 + gi22 * N22);
        }
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
