#include "plate_reissner_mindlin_3p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
PlateReissnerMindlin3p<T>::PlateReissnerMindlin3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlin3p: "
                                    "material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlin3p<T>::strain_matrix(const Patch<T, 2>& /*patch*/,
                                         const BasisDerivs<T, 2>& basis,
                                         const IntrinsicGeometry<T, 2>& ig) const
{
    const Index Q = basis.N().rows();
    const Index n = basis.N().cols();
    Matrix<T> B = Matrix<T>::Zero(5 * Q, 3 * n);


    for (Index q = 0; q < Q; ++q)
    {
        const T Gam1_11 = ig.chr.Gamma[0][0][0](q), Gam1_12 = ig.chr.Gamma[0][0][1](q), Gam1_22 = ig.chr.Gamma[0][1][1](q);
        const T Gam2_11 = ig.chr.Gamma[1][0][0](q), Gam2_12 = ig.chr.Gamma[1][0][1](q), Gam2_22 = ig.chr.Gamma[1][1][1](q);

        for (Index i = 0; i < n; ++i)
        {
            const T Ni   = basis.N()  (q, i);
            const T Ni_u = basis.N_d1(0)(q, i);
            const T Ni_v = basis.N_d1(1)(q, i);

            // B_b = [ 0    N_{i|1} − Γ¹_{11} N_i      −Γ²_{11} N_i            ]   κ_{11}
            //       [ 0    −Γ¹_{22} N_i               N_{i|2} − Γ²_{22} N_i   ]   κ_{22}
            //       [ 0    N_{i|2} − 2 Γ¹_{12} N_i    N_{i|1} − 2 Γ²_{12} N_i ]   2κ_{12}
            B(5*q,     3*i + 1) =  Ni_u - Gam1_11 * Ni;
            B(5*q,     3*i + 2) =       - Gam2_11 * Ni;
            B(5*q + 1, 3*i + 1) =       - Gam1_22 * Ni;
            B(5*q + 1, 3*i + 2) =  Ni_v - Gam2_22 * Ni;
            B(5*q + 2, 3*i + 1) =  Ni_v - T(2) * Gam1_12 * Ni;
            B(5*q + 2, 3*i + 2) =  Ni_u - T(2) * Gam2_12 * Ni;

            // B_s = [ N_{i|1}   N_i    0   ]   γ_1
            //       [ N_{i|2}   0      N_i ]   γ_2
            B(5*q + 3, 3*i    ) =  Ni_u;
            B(5*q + 3, 3*i + 1) =  Ni;
            B(5*q + 4, 3*i    ) =  Ni_v;
            B(5*q + 4, 3*i + 2) =  Ni;
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
                                                     const BasisDerivs<T, 2>& basis,
                                                     const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    const Index Q = basis.N().rows();
    const Index n = basis.N().cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * n);

    for (Index i = 0; i < n; ++i)
    {
        // N_w = [ N_i  0  0 ]
        Nw.col(3*i) = basis.N().col(i);
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T>
PlateReissnerMindlin3p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                 const BasisDerivs<T, 2>& basis,
                                                 const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    const Index Q = basis.N().rows();
    const Index n = basis.N().cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * n);

    for (Index q = 0; q < Q; ++q)
    {
        for (Index i = 0; i < n; ++i) 
        {
            // N_rot = [ 0  N_i  0 ]
            //         [ 0  0   N_i ]
            Nphi(2*q,     3*i + 1) = basis.N()(q, i);
            Nphi(2*q + 1, 3*i + 2) = basis.N()(q, i);
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
