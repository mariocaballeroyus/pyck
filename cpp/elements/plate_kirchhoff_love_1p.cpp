#include "plate_kirchhoff_love_1p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

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
Matrix<T>
PlateKirchhoffLove1p<T>::strain_matrix(const Patch<T, 2>& /*patch*/,
                                       const std::vector<Matrix<T>>& basis,
                                       const IntrinsicGeometry<T, 2>& ig) const
{
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> B(3 * Q, N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = basis[1].col(q);  // (N · 2)
        auto slab2 = basis[2].col(q);  // (N · 3)

        const T Gam1_11 = ig.chr.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ig.chr.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ig.chr.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ig.chr.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ig.chr.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ig.chr.Gamma(1, 1, 1)(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_u_i  = slab1(i * 2 + 0);
            const T N_v_i  = slab1(i * 2 + 1);
            const T N_uu_i = slab2(i * 3 + 0);    // Voigt: (0,0) → 0
            const T N_vv_i = slab2(i * 3 + 1);    // Voigt: (1,1) → 1
            const T N_uv_i = slab2(i * 3 + 2);    // Voigt: (0,1) → 2

            // B_i = [ -N_{i|11}   ]
            //       [ -N_{i|22}   ]
            //       [ -2 N_{i|12} ]
            B(3*q,     i) = -(N_uu_i - Gam1_11 * N_u_i - Gam2_11 * N_v_i);
            B(3*q + 1, i) = -(N_vv_i - Gam1_22 * N_u_i - Gam2_22 * N_v_i);
            B(3*q + 2, i) = -T(2) * (N_uv_i - Gam1_12 * N_u_i - Gam2_12 * N_v_i);
        }
    }
    return B;
}

template <std::floating_point T>
Matrix<T>
PlateKirchhoffLove1p<T>::constitutive_matrix(const IntrinsicGeometry<T, 2>& ig,
                                             Index q) const
{
    // D = D_b. No shear block (normality assumption).
    return material_->bending_voigt(g_inv_voigt(ig, q));
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
PlateKirchhoffLove1p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                   const std::vector<Matrix<T>>& basis,
                                                   const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    // N_w = [ N_i ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> N_w(Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = basis[0].col(q);
        for (Index i = 0; i < N; ++i) {
            N_w(q, i) = slab0(i);
        }
    }
    return N_w;
}

template <std::floating_point T>
Matrix<T>
PlateKirchhoffLove1p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                               const std::vector<Matrix<T>>& basis,
                                               const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    // N_rot = [ -N_{i|1} ]
    //         [ -N_{i|2} ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> N_varphi(2 * Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = basis[1].col(q);
        for (Index i = 0; i < N; ++i) {
            N_varphi(2*q,     i) = -slab1(i * 2 + 0);
            N_varphi(2*q + 1, i) = -slab1(i * 2 + 1);
        }
    }
    return N_varphi;
}

// === Template Instantiations ========================================================

template class PlateKirchhoffLove1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateKirchhoffLove1p<float>;
#endif

} // namespace pyck
