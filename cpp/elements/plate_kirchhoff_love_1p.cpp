#include "plate_kirchhoff_love_1p.hpp"
#include "patch.hpp"
#include "christoffels.hpp"

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
                                       const BasisDerivs<T, 2>& basis,
                                       const LocalFrame<T, 2>& local,
                                       const ChristoffelSymbols<T, 2>& chr) const
{
    const Matrix<T>& N_u  = basis.N_u;
    const Matrix<T>& N_v  = basis.N_v;
    const Matrix<T>& N_uu = basis.N_uu;
    const Matrix<T>& N_uv = basis.N_uv;
    const Matrix<T>& N_vv = basis.N_vv;

    const Index Q = N_u.rows();
    const Index n = N_u.cols();
    Matrix<T> B(3 * Q, n);

    for (Index q = 0; q < Q; ++q)
    {
        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);

        // B_b = [ -N_{i|11}   ]
        //       [ -N_{i|22}   ]
        //       [ -2 N_{i|12} ]
        B.row(3*q    ) = -(N_uu.row(q) - Gam1_11 * N_u.row(q) - Gam2_11 * N_v.row(q));
        B.row(3*q + 1) = -(N_vv.row(q) - Gam1_22 * N_u.row(q) - Gam2_22 * N_v.row(q));
        B.row(3*q + 2) = -T(2) * (N_uv.row(q) - Gam1_12 * N_u.row(q) - Gam2_12 * N_v.row(q));

        // No shear strain component (normality assumption)
    }
    return B;
}

template <std::floating_point T> 
Matrix<T>
PlateKirchhoffLove1p<T>::constitutive_matrix(const LocalFrame<T, 2>& local,
                                             Index q) const
{
    // D = D_b
    // No shear block (normality assumption)
    return material_->bending_voigt(local.g_inv.row(q).transpose());
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T> PlateKirchhoffLove1p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                             const BasisDerivs<T, 2>& basis,
                                                             const LocalFrame<T, 2>& /*local*/,
                                                             const ChristoffelSymbols<T, 2>& /*chr*/) const
{
    // N_w = [ N_i ]
    return basis.N;
}

template <std::floating_point T>
Matrix<T>
PlateKirchhoffLove1p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                               const BasisDerivs<T, 2>& basis,
                                               const LocalFrame<T, 2>& /*local*/,
                                               const ChristoffelSymbols<T, 2>& /*chr*/) const
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

// === Template Instantiations ========================================================

template class PlateKirchhoffLove1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateKirchhoffLove1p<float>;
#endif

} // namespace pyck
