#include "plate_reissner_mindlin_1p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"
#include "laplace_beltrami.hpp"

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
PlateReissnerMindlin1p<T>::constitutive_matrix(const IntrinsicGeometry<T, 2>& ig,
                                               Index q) const
{
    // D = [ D_b   0   ]
    //     [ 0     D_s ]
    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(5, 5);
    D.template topLeftCorner    <3, 3>() = material_->bending_voigt(g_inv_voigt(ig, q));
    D.template bottomRightCorner<2, 2>() = material_->shear_voigt  (g_inv_voigt(ig, q));
    return D;
}

template <std::floating_point T>
void
PlateReissnerMindlin1p<T>::strain_matrix(const Patch<T, 2>& /*patch*/,
                                         const std::vector<Matrix<T>>& basis,
                                         const IntrinsicGeometry<T, 2>& ig) const
{
    auto aux = compute_laplace_grad_aux(ig);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& B = this->B_workspace_;
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    B.setZero(5 * Q, N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = basis[1].col(q);  // (N · 2)
        auto slab2 = basis[2].col(q);  // (N · 3) Voigt: ∂uu, ∂vv, ∂uv
        auto slab3 = basis[3].col(q);  // (N · 4) lex: ∂uuu, ∂uuv, ∂uvv, ∂vvv

        const T Gam1_11 = ig.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ig.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ig.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ig.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ig.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ig.Gamma(1, 1, 1)(q);
        const T G11 = ig.g_inv(0, 0)(q);
        const T G12 = ig.g_inv(0, 1)(q);
        const T G22 = ig.g_inv(1, 1)(q);
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
            const T N_uu_i   = slab2(i * 3 + 0);    // Voigt: (0,0) → 0
            const T N_vv_i   = slab2(i * 3 + 1);    // Voigt: (1,1) → 1
            const T N_uv_i   = slab2(i * 3 + 2);    // Voigt: (0,1) → 2
            const T N_uuu_i  = slab3(i * 4 + 0);    // lex: (0,0,0) → 0
            const T N_uuv_i  = slab3(i * 4 + 1);    // lex: (0,0,1) → 1
            const T N_uvv_i  = slab3(i * 4 + 2);    // lex: (0,1,1) → 2
            const T N_vvv_i  = slab3(i * 4 + 3);    // lex: (1,1,1) → 3

            // B_b = [ -N_{i|11}   ]
            //       [ -N_{i|22}   ]
            //       [ -2 N_{i|12} ]
            B(5*q,     i) = -(N_uu_i - Gam1_11 * N_u_i - Gam2_11 * N_v_i);
            B(5*q + 1, i) = -(N_vv_i - Gam1_22 * N_u_i - Gam2_22 * N_v_i);
            B(5*q + 2, i) = -T(2) * (N_uv_i - Gam1_12 * N_u_i - Gam2_12 * N_v_i);

            // B_s = [ -(K_b/K_s) (Δ_g N_i)_{|1} ]
            //       [ -(K_b/K_s) (Δ_g N_i)_{|2} ]
            B(5*q + 3, i) = -ratio * (
                  G11_d1 * N_uu_i  + T(2)*G12_d1 * N_uv_i + G22_d1 * N_vv_i
                + G11    * N_uuu_i + T(2)*G12    * N_uuv_i + G22    * N_uvv_i
                - c1_d1  * N_u_i   - c2_d1       * N_v_i
                - c1     * N_uu_i  - c2          * N_uv_i);

            B(5*q + 4, i) = -ratio * (
                  G11_d2 * N_uu_i  + T(2)*G12_d2 * N_uv_i + G22_d2 * N_vv_i
                + G11    * N_uuv_i + T(2)*G12    * N_uvv_i + G22    * N_vvv_i
                - c1_d2  * N_u_i   - c2_d2       * N_v_i
                - c1     * N_uv_i  - c2          * N_vv_i);
        }
    }
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
PlateReissnerMindlin1p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                     const std::vector<Matrix<T>>& basis,
                                                     const IntrinsicGeometry<T, 2>& ig) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    Matrix<T>& Nw = this->N_w_workspace_;
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Nw.resize(Q, N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = basis[0].col(q);
        auto slab1 = basis[1].col(q);
        auto slab2 = basis[2].col(q);

        const T gi11 = ig.g_inv(0, 0)(q);
        const T gi12 = ig.g_inv(0, 1)(q);
        const T gi22 = ig.g_inv(1, 1)(q);
        const T Gam1_11 = ig.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ig.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ig.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ig.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ig.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ig.Gamma(1, 1, 1)(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i    = slab0(i);
            const T N_u_i  = slab1(i * 2 + 0);
            const T N_v_i  = slab1(i * 2 + 1);
            const T N_uu_i = slab2(i * 3 + 0);
            const T N_vv_i = slab2(i * 3 + 1);
            const T N_uv_i = slab2(i * 3 + 2);

            const T N11 = N_uu_i - Gam1_11 * N_u_i - Gam2_11 * N_v_i;
            const T N12 = N_uv_i - Gam1_12 * N_u_i - Gam2_12 * N_v_i;
            const T N22 = N_vv_i - Gam1_22 * N_u_i - Gam2_22 * N_v_i;

            // N_w = N_i - (K_b/K_s) Δ_g N_i
            Nw(q, i) = N_i - ratio * (gi11 * N11 + T(2) * gi12 * N12 + gi22 * N22);
        }
    }
}

template <std::floating_point T>
void
PlateReissnerMindlin1p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                 const std::vector<Matrix<T>>& basis,
                                                 const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    // N_rot = [ -N_{i|1} ]
    //         [ -N_{i|2} ]
    Matrix<T>& Nphi = this->N_phi_workspace_;
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Nphi.resize(2 * Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = basis[1].col(q);
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
