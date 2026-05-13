#include "plate_reissner_mindlin_1p.hpp"
#include "patch.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlin1p<T>::PlateReissnerMindlin1p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlin1p: "
                                    "material is null.");
    }
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin1p<T>::displacement_shape_matrix(const Patch<T, 2>& patch, 
                                                     const BasisDerivs<T, 2>& basis, 
                                                     const LocalFrame<T, 2>& local) const
{
    auto chr = patch.eval_christoffel(local);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nw(Q, n);

    // N_w = [ N_i - (K_b/K_s) Δ_g N_i ]
    for (Index q = 0; q < Q; ++q)
    {
        const T gi11 = local.g_inv(q, 0);
        const T gi12 = local.g_inv(q, 1);
        const T gi22 = local.g_inv(q, 2);

        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);

        const auto N11 = basis.N_uu.row(q) - Gam1_11 * basis.N_u.row(q) - Gam2_11 * basis.N_v.row(q);
        const auto N12 = basis.N_uv.row(q) - Gam1_12 * basis.N_u.row(q) - Gam2_12 * basis.N_v.row(q);
        const auto N22 = basis.N_vv.row(q) - Gam1_22 * basis.N_u.row(q) - Gam2_22 * basis.N_v.row(q);
        
        Nw.row(q) = basis.N.row(q) - ratio * (gi11 * N11 + T(2) * gi12 * N12 + gi22 * N22);
    }
    return Nw;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin1p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/, 
                                                 const BasisDerivs<T, 2>& basis, 
                                                 const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N_u.rows();
    const Index n = basis.N_u.cols();
    Matrix<T> Nphi(2 * Q, n);

    // N_rot = [ -N_{i|1} ]
    //         [ -N_{i|2} ]
    for (Index q = 0; q < Q; ++q)
    {
        Nphi.row(2*q    ) = -basis.N_u.row(q);
        Nphi.row(2*q + 1) = -basis.N_v.row(q);
    }
    return Nphi;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin1p<T>::bending_constitutive_matrix(const LocalFrame<T, 2>& local, 
                                                       Index q) const
{
    // D_b = (g^{-1})^T D_{ps} (g^{-1})
    return material_->bending_voigt(local.g_inv.row(q).transpose());
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin1p<T>::shear_constitutive_matrix(const LocalFrame<T, 2>& local, 
                                                     Index q) const
{
    // D_s = (g^{-1})^T D_{ps} (g^{-1})
    return material_->shear_voigt(local.g_inv.row(q).transpose());
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin1p<T>::bending_strain_matrix(const Patch<T, 2>& patch, 
                                                 const BasisDerivs<T, 2>& basis, 
                                                 const LocalFrame<T, 2>& local) const
{
    auto chr = patch.eval_christoffel(local);
    const Index Q = basis.N_u.rows();
    const Index n = basis.N_u.cols();
    Matrix<T> Bb(3 * Q, n);

    // B_b = [ -N_{i|11}    ]
    //       [ -N_{i|22}    ]
    //       [ -2 N_{i|12}  ]
    for (Index q = 0; q < Q; ++q)
    {
        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);

        Bb.row(3*q    ) = -(basis.N_uu.row(q) - Gam1_11 * basis.N_u.row(q) - Gam2_11 * basis.N_v.row(q));
        Bb.row(3*q + 1) = -(basis.N_vv.row(q) - Gam1_22 * basis.N_u.row(q) - Gam2_22 * basis.N_v.row(q));
        Bb.row(3*q + 2) = -T(2) * (basis.N_uv.row(q) - Gam1_12 * basis.N_u.row(q) - Gam2_12 * basis.N_v.row(q));
    }
    return Bb;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin1p<T>::shear_strain_matrix(const Patch<T, 2>& patch, 
                                               const BasisDerivs<T, 2>& basis, 
                                               const LocalFrame<T, 2>& local) const
{
    auto chr = patch.eval_christoffel(local);
    auto aux = compute_laplace_grad_aux(local, chr);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Bs(2 * Q, n);

    // B_s = [ -(K_b/K_s) (Δ_g N_i)_{|1} ]
    //       [ -(K_b/K_s) (Δ_g N_i)_{|2} ]
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

        Bs.row(2*q    ) = -ratio * (
            G11_d1 * basis.N_uu.row(q) + T(2)*G12_d1 * basis.N_uv.row(q) + G22_d1 * basis.N_vv.row(q)
          + G11    * basis.N_uuu.row(q) + T(2)*G12    * basis.N_uuv.row(q) + G22    * basis.N_uvv.row(q)
          - c1_d1  * basis.N_u.row(q)   - c2_d1       * basis.N_v.row(q)
          - c1     * basis.N_uu.row(q)  - c2          * basis.N_uv.row(q));

        Bs.row(2*q + 1) = -ratio * (
            G11_d2 * basis.N_uu.row(q) + T(2)*G12_d2 * basis.N_uv.row(q) + G22_d2 * basis.N_vv.row(q)
          + G11    * basis.N_uuv.row(q) + T(2)*G12    * basis.N_uvv.row(q) + G22    * basis.N_vvv.row(q)
          - c1_d2  * basis.N_u.row(q)   - c2_d2       * basis.N_v.row(q)
          - c1     * basis.N_uv.row(q)  - c2          * basis.N_vv.row(q));
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin1p<float>;
#endif

} // namespace pyck
