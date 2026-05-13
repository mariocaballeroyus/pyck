#include "plate_reissner_mindlin_displ_3p.hpp"
#include "patch.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlinDispl3p<T>::PlateReissnerMindlinDispl3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlinDispl3p: "
                                    "material is null.");
    }
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlinDispl3p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/, 
                                                          const BasisDerivs<T, 2>& basis, 
                                                          const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * n);

    // N_w = [ N_i  N_i  N_i ]
    for (Index i = 0; i < n; ++i) 
    {
        Nw.col(3 * i    ) = basis.N.col(i);
        Nw.col(3 * i + 1) = basis.N.col(i);
        Nw.col(3 * i + 2) = basis.N.col(i);
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl3p<T>::rotation_shape_matrix(
    const Patch<T, 2>& /*patch*/, const BasisDerivs<T, 2>& basis, const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * n);

    // N_rot = [ -N_{i|1}   0         -N_{i|1} ]
    //         [ -N_{i|2}  -N_{i|2}    0       ]
    for (Index q = 0; q < Q; ++q) 
    {
        for (Index i = 0; i < n; ++i) 
        {
            Nphi(2 * q,     3 * i)     = -basis.N_u(q, i);
            Nphi(2 * q,     3 * i + 2) = -basis.N_u(q, i);
            Nphi(2 * q + 1, 3 * i)     = -basis.N_v(q, i);
            Nphi(2 * q + 1, 3 * i + 1) = -basis.N_v(q, i);
        }
    }
    return Nphi;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlinDispl3p<T>::bending_constitutive_matrix(const LocalFrame<T, 2>& local, 
                                                            Index q) const
{
    return material_->bending_voigt(local.g_inv.row(q).transpose());
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlinDispl3p<T>::shear_constitutive_matrix(const LocalFrame<T, 2>& local, 
                                                          Index q) const
{
    return material_->shear_voigt(local.g_inv.row(q).transpose());
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlinDispl3p<T>::bending_strain_matrix(const Patch<T, 2>& patch, 
                                                      const BasisDerivs<T, 2>& basis, 
                                                      const LocalFrame<T, 2>& local) const
{
    auto chr = patch.eval_christoffel(local);
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Bb = Matrix<T>::Zero(3 * Q, 3 * n);

    // B_b = [ -N_{i|11}     0           -N_{i|11} ]
    //       [ -N_{i|22}    -N_{i|22}     0        ]
    //       [ -2 N_{i|12}  -N_{i|12}    -N_{i|12} ]
    for (Index q = 0; q < Q; ++q) 
    {
        const T Gam1_11 = chr.G1_11(q), Gam1_12 = chr.G1_12(q), Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q), Gam2_12 = chr.G2_12(q), Gam2_22 = chr.G2_22(q);
        const auto N11 = basis.N_uu.row(q) - Gam1_11 * basis.N_u.row(q) - Gam2_11 * basis.N_v.row(q);
        const auto N12 = basis.N_uv.row(q) - Gam1_12 * basis.N_u.row(q) - Gam2_12 * basis.N_v.row(q);
        const auto N22 = basis.N_vv.row(q) - Gam1_22 * basis.N_u.row(q) - Gam2_22 * basis.N_v.row(q);

        for (Index i = 0; i < n; ++i) 
        {
            Bb(3 * q,     3 * i)     = -N11(i);
            Bb(3 * q,     3 * i + 2) = -N11(i);
            Bb(3 * q + 1, 3 * i)     = -N22(i);
            Bb(3 * q + 1, 3 * i + 1) = -N22(i);
            Bb(3 * q + 2, 3 * i)     = -T(2) * N12(i);
            Bb(3 * q + 2, 3 * i + 1) = -N12(i);
            Bb(3 * q + 2, 3 * i + 2) = -N12(i);
        }
    }
    return Bb;
}

template <std::floating_point T> Matrix<T> 
PlateReissnerMindlinDispl3p<T>::shear_strain_matrix(const Patch<T, 2>& /*patch*/, 
                                                    const BasisDerivs<T, 2>& basis, 
                                                    const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Bs = Matrix<T>::Zero(2 * Q, 3 * n);

    // B_s = [ 0   N_{i|1}   0       ]
    //       [ 0   0         N_{i|2} ]
    for (Index q = 0; q < Q; ++q) 
    {
        for (Index i = 0; i < n; ++i) 
        {
            Bs(2 * q,     3 * i + 1) = basis.N_u(q, i);
            Bs(2 * q + 1, 3 * i + 2) = basis.N_v(q, i);
        }
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlinDispl3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlinDispl3p<float>;
#endif

} // namespace pyck
