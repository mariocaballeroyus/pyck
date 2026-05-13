#include "plate_reissner_mindlin_3p.hpp"
#include "patch.hpp"
#include "christoffels.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlin3p<T>::PlateReissnerMindlin3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlin3p: "
                                    "material is null.");
    }
}

template <std::floating_point T> 
Matrix<T> 
PlateReissnerMindlin3p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                     const BasisDerivs<T, 2>& basis,
                                                     const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * n);

    // N_w = [ N_i  0  0 ]
    for (Index i = 0; i < n; ++i)
    {
        Nw.col(3*i) = basis.N.col(i);
    }
    return Nw;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin3p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                 const BasisDerivs<T, 2>& basis,
                                                 const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * n);

    // N_rot = [ 0  N_i  0 ]
    //         [ 0  0   N_i ]
    for (Index q = 0; q < Q; ++q)
    {
        for (Index i = 0; i < n; ++i) 
        {
            Nphi(2*q,     3*i + 1) = basis.N(q, i);
            Nphi(2*q + 1, 3*i + 2) = basis.N(q, i);
        }
    }
    return Nphi;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin3p<T>::bending_constitutive_matrix(const LocalFrame<T, 2>& local,
                                                       Index q) const
{
    // D_b = g^{-T} D_{ps} g^{-1}
    return material_->bending_voigt(local.g_inv.row(q).transpose());
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin3p<T>::shear_constitutive_matrix(const LocalFrame<T, 2>& local,
                                                     Index q) const
{
    // D_s = g^{-T} D_{ps} g^{-1}
    return material_->shear_voigt(local.g_inv.row(q).transpose());
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin3p<T>::bending_strain_matrix(const Patch<T, 2>& patch,
                                                 const BasisDerivs<T, 2>& basis,
                                                 const LocalFrame<T, 2>& local) const
{
    auto chr = eval_christoffel(local);
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Bb = Matrix<T>::Zero(3 * Q, 3 * n);

    // B_b = [ 0  N_{i|1} − Γ¹_{11} N_i      −Γ²_{11} N_i            ]
    //       [ 0  −Γ¹_{22} N_i               N_{i|2} − Γ²_{22} N_i   ]
    //       [ 0  N_{i|2} − 2 Γ¹_{12} N_i    N_{i|1} − 2 Γ²_{12} N_i ]
    for (Index q = 0; q < Q; ++q)
    {
        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);

        for (Index i = 0; i < n; ++i) 
        {
            const T Ni   = basis.N  (q, i);
            const T Ni_u = basis.N_u(q, i);
            const T Ni_v = basis.N_v(q, i);

            Bb(3*q,     3*i + 1) =  Ni_u - Gam1_11 * Ni;
            Bb(3*q,     3*i + 2) =       - Gam2_11 * Ni;
            Bb(3*q + 1, 3*i + 1) =       - Gam1_22 * Ni;
            Bb(3*q + 1, 3*i + 2) =  Ni_v - Gam2_22 * Ni;
            Bb(3*q + 2, 3*i + 1) =  Ni_v - T(2) * Gam1_12 * Ni;
            Bb(3*q + 2, 3*i + 2) =  Ni_u - T(2) * Gam2_12 * Ni;
        }
    }
    return Bb;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlin3p<T>::shear_strain_matrix(const Patch<T, 2>& /*patch*/,
                                               const BasisDerivs<T, 2>& basis,
                                               const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Bs = Matrix<T>::Zero(2 * Q, 3 * n);

    // B_s = [ N_{i|1}  N_i   0  ]
    //       [ N_{i|2}  0    N_i ]
    for (Index q = 0; q < Q; ++q)
    {
        for (Index i = 0; i < n; ++i) 
        {
            Bs(2*q,     3*i    ) = basis.N_u(q, i);
            Bs(2*q,     3*i + 1) = basis.N  (q, i);
            Bs(2*q + 1, 3*i    ) = basis.N_v(q, i);
            Bs(2*q + 1, 3*i + 2) = basis.N  (q, i);
        }
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin3p<float>;
#endif

} // namespace pyck
