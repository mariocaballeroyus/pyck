#include "plate_reissner_mindlin_displ_3p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlinDispl3p<T>::PlateReissnerMindlinDispl3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlinDispl3p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlinDispl3p<T>::strain_matrix(const Patch<T, 2>& /*patch*/,
                                              const std::vector<Matrix<T>>& basis,
                                              const IntrinsicGeometry<T, 2>& ig) const
{
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> B = Matrix<T>::Zero(5 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = basis[1].col(q);
        auto slab2 = basis[2].col(q);

        const T Gam1_11 = ig.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ig.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ig.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ig.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ig.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ig.Gamma(1, 1, 1)(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_u_i  = slab1(i * 2 + 0);
            const T N_v_i  = slab1(i * 2 + 1);
            const T N_uu_i = slab2(i * 3 + 0);    // Voigt: (0,0) → 0
            const T N_vv_i = slab2(i * 3 + 1);    // Voigt: (1,1) → 1
            const T N_uv_i = slab2(i * 3 + 2);    // Voigt: (0,1) → 2

            const T N11 = N_uu_i - Gam1_11 * N_u_i - Gam2_11 * N_v_i;
            const T N12 = N_uv_i - Gam1_12 * N_u_i - Gam2_12 * N_v_i;
            const T N22 = N_vv_i - Gam1_22 * N_u_i - Gam2_22 * N_v_i;

            // B_b = [ -N_{i|11}     0           -N_{i|11}      ]   κ_{11}
            //       [ -N_{i|22}    -N_{i|22}     0             ]   κ_{22}
            //       [ -2 N_{i|12}  -N_{i|12}    -N_{i|12}      ]   2κ_{12}
            B(5*q,     3*i    ) = -N11;
            B(5*q,     3*i + 2) = -N11;
            B(5*q + 1, 3*i    ) = -N22;
            B(5*q + 1, 3*i + 1) = -N22;
            B(5*q + 2, 3*i    ) = -T(2) * N12;
            B(5*q + 2, 3*i + 1) = -N12;
            B(5*q + 2, 3*i + 2) = -N12;

            // B_s = [ 0   N_{i|1}  0       ]   γ_1
            //       [ 0   0        N_{i|2} ]   γ_2
            B(5*q + 3, 3*i + 1) = N_u_i;
            B(5*q + 4, 3*i + 2) = N_v_i;
        }
    }
    return B;
}

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlinDispl3p<T>::constitutive_matrix(const IntrinsicGeometry<T, 2>& ig,
                                                    Index q) const
{
    // D = [ D_b   0   ]
    //     [ 0     D_s ]
    Matrix<T> D = Matrix<T>::Zero(5, 5);
    D.template topLeftCorner    <3, 3>() = material_->bending_voigt(g_inv_voigt(ig, q));
    D.template bottomRightCorner<2, 2>() = material_->shear_voigt  (g_inv_voigt(ig, q));
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlinDispl3p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                          const std::vector<Matrix<T>>& basis,
                                                          const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    // N_w = [ N_i  N_i  N_i ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = basis[0].col(q);
        for (Index i = 0; i < N; ++i) {
            const T N_i = slab0(i);
            Nw(q, 3*i    ) = N_i;
            Nw(q, 3*i + 1) = N_i;
            Nw(q, 3*i + 2) = N_i;
        }
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlinDispl3p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                      const std::vector<Matrix<T>>& basis,
                                                      const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    // N_rot = [ -N_{i|1}   0         -N_{i|1} ]
    //         [ -N_{i|2}  -N_{i|2}    0       ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = basis[1].col(q);
        for (Index i = 0; i < N; ++i) {
            const T N_u_i = slab1(i * 2 + 0);
            const T N_v_i = slab1(i * 2 + 1);
            Nphi(2*q,     3*i    ) = -N_u_i;
            Nphi(2*q,     3*i + 2) = -N_u_i;
            Nphi(2*q + 1, 3*i    ) = -N_v_i;
            Nphi(2*q + 1, 3*i + 1) = -N_v_i;
        }
    }
    return Nphi;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlinDispl3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlinDispl3p<float>;
#endif

} // namespace pyck
