#include "plate_reissner_mindlin_displ_2p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"
#include "laplace_beltrami.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
PlateReissnerMindlinDispl2p<T>::PlateReissnerMindlinDispl2p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlinDispl2p: "
                                   "material is null.");
    }
}

// === Matrix Operators ===============================================================


template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::strain_matrix(
    const Patch<T, 2>& /*patch*/, const BasisDerivs<T, 2>& basis, const IntrinsicGeometry<T, 2>& ig) const
{
    auto aux = compute_laplace_grad_aux(ig);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    const Index Q = basis.N().rows();
    const Index n = basis.N().cols();
    Matrix<T> B = Matrix<T>::Zero(5 * Q, 2 * n);

    for (Index q = 0; q < Q; ++q)
    {
        const T Gam1_11 = ig.chr.Gamma[0][0][0](q), Gam1_12 = ig.chr.Gamma[0][0][1](q), Gam1_22 = ig.chr.Gamma[0][1][1](q);
        const T Gam2_11 = ig.chr.Gamma[1][0][0](q), Gam2_12 = ig.chr.Gamma[1][0][1](q), Gam2_22 = ig.chr.Gamma[1][1][1](q);
        const auto N11 = basis.N_d2(0, 0).row(q) - Gam1_11 * basis.N_d1(0).row(q) - Gam2_11 * basis.N_d1(1).row(q);
        const auto N12 = basis.N_d2(0, 1).row(q) - Gam1_12 * basis.N_d1(0).row(q) - Gam2_12 * basis.N_d1(1).row(q);
        const auto N22 = basis.N_d2(1, 1).row(q) - Gam1_22 * basis.N_d1(0).row(q) - Gam2_22 * basis.N_d1(1).row(q);

        const T G11 = ig.g_inv[0][0](q);
        const T G12 = ig.g_inv[0][1](q);
        const T G22 = ig.g_inv[1][1](q);
        const T G11_d1 = aux.G_inv_d[0][0][0](q), G12_d1 = aux.G_inv_d[0][1][0](q), G22_d1 = aux.G_inv_d[1][1][0](q);
        const T G11_d2 = aux.G_inv_d[0][0][1](q), G12_d2 = aux.G_inv_d[0][1][1](q), G22_d2 = aux.G_inv_d[1][1][1](q);
        const T c1 = aux.c[0](q), c2 = aux.c[1](q);
        const T c1_d1 = aux.c_d[0][0](q), c2_d1 = aux.c_d[1][0](q);
        const T c1_d2 = aux.c_d[0][1](q), c2_d2 = aux.c_d[1][1](q);

        const auto lap_1 = ratio * (
              G11_d1 * basis.N_d2(0, 0).row(q)  + T(2)*G12_d1 * basis.N_d2(0, 1).row(q) + G22_d1 * basis.N_d2(1, 1).row(q)
            + G11    * basis.N_d3(0, 0, 0).row(q) + T(2)*G12    * basis.N_d3(0, 0, 1).row(q) + G22   * basis.N_d3(0, 1, 1).row(q)
            - c1_d1  * basis.N_d1(0).row(q)   - c2_d1       * basis.N_d1(1).row(q)
            - c1     * basis.N_d2(0, 0).row(q)  - c2          * basis.N_d2(0, 1).row(q));
        const auto lap_2 = ratio * (
              G11_d2 * basis.N_d2(0, 0).row(q)  + T(2)*G12_d2 * basis.N_d2(0, 1).row(q) + G22_d2 * basis.N_d2(1, 1).row(q)
            + G11    * basis.N_d3(0, 0, 1).row(q) + T(2)*G12    * basis.N_d3(0, 1, 1).row(q) + G22   * basis.N_d3(1, 1, 1).row(q)
            - c1_d2  * basis.N_d1(0).row(q)   - c2_d2       * basis.N_d1(1).row(q)
            - c1     * basis.N_d2(0, 1).row(q)  - c2          * basis.N_d2(1, 1).row(q));

        for (Index i = 0; i < n; ++i)
        {
            // B_b = [ -N_{i|11}    N_{i|12}          ]   κ_{11}
            //       [ -N_{i|22}   -N_{i|12}          ]   κ_{22}
            //       [ -2 N_{i|12}  N_{i|22}−N_{i|11} ]   2κ_{12}
            B(5 * q,     2 * i    ) = -N11(i);
            B(5 * q,     2 * i + 1) =  N12(i);
            B(5 * q + 1, 2 * i    ) = -N22(i);
            B(5 * q + 1, 2 * i + 1) = -N12(i);
            B(5 * q + 2, 2 * i    ) = -T(2) * N12(i);
            B(5 * q + 2, 2 * i + 1) =  N22(i) - N11(i);

            // B_s = [ -(K_b/K_s) (Δ_g N_i)_{|1}    N_{i|2} ]   γ_1
            //       [ -(K_b/K_s) (Δ_g N_i)_{|2}   -N_{i|1} ]   γ_2
            B(5 * q + 3, 2 * i    ) = -lap_1(i);
            B(5 * q + 3, 2 * i + 1) =  basis.N_d1(1)(q, i);
            B(5 * q + 4, 2 * i    ) = -lap_2(i);
            B(5 * q + 4, 2 * i + 1) = -basis.N_d1(0)(q, i);
        }
    }
    return B;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::constitutive_matrix(
    const IntrinsicGeometry<T, 2>& ig, Index q) const
{
    // D = [ D_b  0   ]
    //     [  0   D_s ]
    Matrix<T> D = Matrix<T>::Zero(5, 5);
    D.template topLeftCorner    <3, 3>() = material_->bending_voigt(g_inv_voigt(ig, q));
    D.template bottomRightCorner<2, 2>() = material_->shear_voigt  (g_inv_voigt(ig, q));
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::displacement_shape_matrix(const Patch<T, 2>& patch,
                                                                    const BasisDerivs<T, 2>& basis,
                                                                    const IntrinsicGeometry<T, 2>& ig) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    const Index Q = basis.N().rows();
    const Index n = basis.N().cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 2 * n);


    for (Index q = 0; q < Q; ++q)
    {
        const T gi11 = ig.g_inv[0][0](q);
        const T gi12 = ig.g_inv[0][1](q);
        const T gi22 = ig.g_inv[1][1](q);

        const T Gam1_11 = ig.chr.Gamma[0][0][0](q), Gam1_12 = ig.chr.Gamma[0][0][1](q), Gam1_22 = ig.chr.Gamma[0][1][1](q);
        const T Gam2_11 = ig.chr.Gamma[1][0][0](q), Gam2_12 = ig.chr.Gamma[1][0][1](q), Gam2_22 = ig.chr.Gamma[1][1][1](q);

        const auto N11 = basis.N_d2(0, 0).row(q) - Gam1_11 * basis.N_d1(0).row(q) - Gam2_11 * basis.N_d1(1).row(q);
        const auto N12 = basis.N_d2(0, 1).row(q) - Gam1_12 * basis.N_d1(0).row(q) - Gam2_12 * basis.N_d1(1).row(q);
        const auto N22 = basis.N_d2(1, 1).row(q) - Gam1_22 * basis.N_d1(0).row(q) - Gam2_22 * basis.N_d1(1).row(q);
        const auto Ntilde = basis.N().row(q) - ratio * (gi11 * N11 + T(2) * gi12 * N12 + gi22 * N22);

        for (Index i = 0; i < n; ++i) 
        {
            // N_w = [ N_i − (K_b/K_s) Δ_g N_i   0 ]
            Nw(q, 2 * i) = Ntilde(i);
        }
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlinDispl2p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                      const BasisDerivs<T, 2>& basis,
                                                      const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    const Index Q = basis.N().rows();
    const Index n = basis.N().cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 2 * n);

    for (Index q = 0; q < Q; ++q) 
    {
        for (Index i = 0; i < n; ++i) 
        {
            // N_rot = [ -N_{i|1}   N_{i|2} ]
            //         [ -N_{i|2}  -N_{i|1} ]
            Nphi(2 * q,     2 * i)     = -basis.N_d1(0)(q, i);
            Nphi(2 * q,     2 * i + 1) =  basis.N_d1(1)(q, i);
            Nphi(2 * q + 1, 2 * i)     = -basis.N_d1(1)(q, i);
            Nphi(2 * q + 1, 2 * i + 1) = -basis.N_d1(0)(q, i);
        }
    }
    return Nphi;
}


// === Template Instantiations ========================================================

template class PlateReissnerMindlinDispl2p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlinDispl2p<float>;
#endif

} // namespace pyck
