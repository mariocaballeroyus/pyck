#include "plate_reissner_mindlin_3p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlin3p<T>::PlateReissnerMindlin3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlin3p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlin3p<T>::strain_matrix(const Patch<T, 2>& /*patch*/,
                                         const std::vector<Matrix<T>>& basis,
                                         const IntrinsicGeometry<T, 2>& ig) const
{
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> B = Matrix<T>::Zero(5 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = basis[0].col(q);
        auto slab1 = basis[1].col(q);

        const T Gam1_11 = ig.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ig.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ig.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ig.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ig.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ig.Gamma(1, 1, 1)(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i   = slab0(i);
            const T N_u_i = slab1(i * 2 + 0);
            const T N_v_i = slab1(i * 2 + 1);

            // B_b = [ 0    N_{i|1} − Γ¹_{11} N_i      −Γ²_{11} N_i           ]   κ_{11}
            //       [ 0    −Γ¹_{22} N_i               N_{i|2} − Γ²_{22} N_i  ]   κ_{22}
            //       [ 0    N_{i|2} − 2 Γ¹_{12} N_i    N_{i|1} − 2 Γ²_{12} N_i ]  2κ_{12}
            B(5*q,     3*i + 1) =  N_u_i - Gam1_11 * N_i;
            B(5*q,     3*i + 2) =        - Gam2_11 * N_i;
            B(5*q + 1, 3*i + 1) =        - Gam1_22 * N_i;
            B(5*q + 1, 3*i + 2) =  N_v_i - Gam2_22 * N_i;
            B(5*q + 2, 3*i + 1) =  N_v_i - T(2) * Gam1_12 * N_i;
            B(5*q + 2, 3*i + 2) =  N_u_i - T(2) * Gam2_12 * N_i;

            // B_s = [ N_{i|1}   N_i    0   ]   γ_1
            //       [ N_{i|2}   0      N_i ]   γ_2
            B(5*q + 3, 3*i    ) =  N_u_i;
            B(5*q + 3, 3*i + 1) =  N_i;
            B(5*q + 4, 3*i    ) =  N_v_i;
            B(5*q + 4, 3*i + 2) =  N_i;
        }
    }
    return B;
}

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlin3p<T>::constitutive_matrix(const IntrinsicGeometry<T, 2>& ig,
                                               Index q) const
{
    // D = [ D_b   0  ]
    //     [  0   D_s ]
    Matrix<T> D = Matrix<T>::Zero(5, 5);
    D.template topLeftCorner    <3, 3>() = material_->bending_voigt(g_inv_voigt(ig, q));
    D.template bottomRightCorner<2, 2>() = material_->shear_voigt  (g_inv_voigt(ig, q));
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlin3p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                     const std::vector<Matrix<T>>& basis,
                                                     const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    // N_w = [ N_i  0  0 ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = basis[0].col(q);
        for (Index i = 0; i < N; ++i) {
            Nw(q, 3*i) = slab0(i);
        }
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlin3p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                 const std::vector<Matrix<T>>& basis,
                                                 const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    // N_rot = [ 0  N_i  0   ]
    //         [ 0  0    N_i ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = basis[0].col(q);
        for (Index i = 0; i < N; ++i) {
            Nphi(2*q,     3*i + 1) = slab0(i);
            Nphi(2*q + 1, 3*i + 2) = slab0(i);
        }
    }
    return Nphi;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin3p<float>;
#endif

} // namespace pyck
