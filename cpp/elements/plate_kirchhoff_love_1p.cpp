#include "plate_kirchhoff_love_1p.hpp"
#include "patch.hpp"

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

template <std::floating_point T>
Matrix<T> PlateKirchhoffLove1p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                             const BasisDerivs<T, 2>& basis,
                                                             const LocalFrame<T, 2>& /*local*/) const
{
    // N_w = [ N_i ]
    return basis.N;
}

template <std::floating_point T> Matrix<T> 
PlateKirchhoffLove1p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                               const BasisDerivs<T, 2>& basis,
                                               const LocalFrame<T, 2>& /*local*/) const
{
    const Matrix<T>& N_u = basis.N_u;
    const Matrix<T>& N_v = basis.N_v;
    const Index Q = N_u.rows();
    const Index n = N_u.cols();
    Matrix<T> N_varphi(2 * Q, n);

    // N_rot = [ -N_{i|1} ]
    //         [ -N_{i|2} ]
    for (Index q = 0; q < Q; ++q)
    {
        N_varphi.row(2*q    ) = -N_u.row(q);
        N_varphi.row(2*q + 1) = -N_v.row(q);
    }
    return N_varphi;
}

template <std::floating_point T> Matrix<T> 
PlateKirchhoffLove1p<T>::bending_constitutive_matrix(const LocalFrame<T, 2>& local, 
                                                     Index q) const
{
    // D_b = (g^{-1})^T D (g^{-1})
    return material_->bending_voigt(local.g_inv.row(q).transpose());
}

template <std::floating_point T> Matrix<T> 
PlateKirchhoffLove1p<T>::bending_strain_matrix(const Patch<T, 2>& patch,
                                               const BasisDerivs<T, 2>& basis,
                                               const LocalFrame<T, 2>& local) const
{
    auto chr = patch.eval_christoffel(local);
    const Matrix<T>& N_u  = basis.N_u;
    const Matrix<T>& N_v  = basis.N_v;
    const Matrix<T>& N_uu = basis.N_uu;
    const Matrix<T>& N_uv = basis.N_uv;
    const Matrix<T>& N_vv = basis.N_vv;
    const Index Q = N_u.rows();
    const Index n = N_u.cols();
    Matrix<T> Bb(3 * Q, n);

    // B_b = [ -N_{i|11}   ]
    //       [ -N_{i|22}   ]
    //       [ -2 N_{i|12} ]
    for (Index q = 0; q < Q; ++q)
    {
        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);

        Bb.row(3*q    ) = -(N_uu.row(q) - Gam1_11 * N_u.row(q) - Gam2_11 * N_v.row(q));
        Bb.row(3*q + 1) = -(N_vv.row(q) - Gam1_22 * N_u.row(q) - Gam2_22 * N_v.row(q));
        Bb.row(3*q + 2) = -T(2) * (N_uv.row(q) - Gam1_12 * N_u.row(q) - Gam2_12 * N_v.row(q));
    }
    return Bb;
}

template <std::floating_point T> Matrix<T> 
PlateKirchhoffLove1p<T>::shear_strain_matrix(const Patch<T, 2>& /*patch*/,
                                             const BasisDerivs<T, 2>& basis,
                                             const LocalFrame<T, 2>& /*local*/) const
{
    // Normality assumption: zero transverse shear strain
    return Matrix<T>::Zero(0, basis.N.cols());
}

template <std::floating_point T> Matrix<T>
PlateKirchhoffLove1p<T>::transverse_shear_matrix(const Patch<T, 2>& patch,
                                                 const BasisDerivs<T, 2>& basis,
                                                 const LocalFrame<T, 2>& local) const
{
    auto chr = patch.eval_christoffel(local);
    auto aux = compute_laplace_grad_aux(local, chr);
    const T Kb = material_->bending_stiffness();
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nq(2 * Q, n);

    // N_q = [ -K_b ( g^{11} (Δ_g N_i)_{|1} + g^{12} (Δ_g N_i)_{|2} ) ]
    //       [ -K_b ( g^{12} (Δ_g N_i)_{|1} + g^{22} (Δ_g N_i)_{|2} ) ]
    for (Index q = 0; q < Q; ++q) 
    {
        const T G11 = local.g_inv(q, 0);
        const T G12 = local.g_inv(q, 1);
        const T G22 = local.g_inv(q, 2);
        const T G11_d1 = aux.G11_d1(q), G12_d1 = aux.G12_d1(q), G22_d1 = aux.G22_d1(q);
        const T G11_d2 = aux.G11_d2(q), G12_d2 = aux.G12_d2(q), G22_d2 = aux.G22_d2(q);
        const T c1 = aux.c1(q), c2 = aux.c2(q);
        const T c1_d1 = aux.c1_d1(q), c2_d1 = aux.c2_d1(q);
        const T c1_d2 = aux.c1_d2(q), c2_d2 = aux.c2_d2(q);

        // ∂_β (Δ_g N)
        const auto lap_d1 = (
              G11_d1 * basis.N_uu.row(q) + T(2)*G12_d1 * basis.N_uv.row(q) + G22_d1 * basis.N_vv.row(q)
            + G11    * basis.N_uuu.row(q) + T(2)*G12   * basis.N_uuv.row(q) + G22    * basis.N_uvv.row(q)
            - c1_d1  * basis.N_u.row(q)   - c2_d1      * basis.N_v.row(q)
            - c1     * basis.N_uu.row(q)  - c2         * basis.N_uv.row(q));
        const auto lap_d2 = (
              G11_d2 * basis.N_uu.row(q) + T(2)*G12_d2 * basis.N_uv.row(q) + G22_d2 * basis.N_vv.row(q)
            + G11    * basis.N_uuv.row(q) + T(2)*G12   * basis.N_uvv.row(q) + G22    * basis.N_vvv.row(q)
            - c1_d2  * basis.N_u.row(q)   - c2_d2      * basis.N_v.row(q)
            - c1     * basis.N_uv.row(q)  - c2         * basis.N_vv.row(q));

        Nq.row(2*q    ) = -Kb * (G11 * lap_d1 + G12 * lap_d2);
        Nq.row(2*q + 1) = -Kb * (G12 * lap_d1 + G22 * lap_d2);
    }
    return Nq;
}

// === Template Instantiations ========================================================

template class PlateKirchhoffLove1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateKirchhoffLove1p<float>;
#endif

} // namespace pyck
