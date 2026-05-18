#include "plate_reissner_mindlin_displ_3p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
PlateReissnerMindlinDispl3p<T>::PlateReissnerMindlinDispl3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlinDispl3p: "
                                    "material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T> Matrix<T>
PlateReissnerMindlinDispl3p<T>::strain_matrix(const Patch<T, 2>& /*patch*/,
                                              const BasisDerivs<T, 2>& basis,
                                              const IntrinsicGeometry<T, 2>& ig) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> B = Matrix<T>::Zero(5 * Q, 3 * n);


    for (Index q = 0; q < Q; ++q)
    {
        const T Gam1_11 = ig.chr.Gamma[0][0][0](q), Gam1_12 = ig.chr.Gamma[0][0][1](q), Gam1_22 = ig.chr.Gamma[0][1][1](q);
        const T Gam2_11 = ig.chr.Gamma[1][0][0](q), Gam2_12 = ig.chr.Gamma[1][0][1](q), Gam2_22 = ig.chr.Gamma[1][1][1](q);
        const auto N11 = basis.N_d2[0][0].row(q) - Gam1_11 * basis.N_d1[0].row(q) - Gam2_11 * basis.N_d1[1].row(q);
        const auto N12 = basis.N_d2[0][1].row(q) - Gam1_12 * basis.N_d1[0].row(q) - Gam2_12 * basis.N_d1[1].row(q);
        const auto N22 = basis.N_d2[1][1].row(q) - Gam1_22 * basis.N_d1[0].row(q) - Gam2_22 * basis.N_d1[1].row(q);

        for (Index i = 0; i < n; ++i)
        {
            // B_b = [ -N_{i|11}     0           -N_{i|11}      ]   κ_{11}
            //       [ -N_{i|22}    -N_{i|22}     0             ]   κ_{22}
            //       [ -2 N_{i|12}  -N_{i|12}    -N_{i|12}      ]   2κ_{12}
            B(5 * q,     3 * i    ) = -N11(i);
            B(5 * q,     3 * i + 2) = -N11(i);
            B(5 * q + 1, 3 * i    ) = -N22(i);
            B(5 * q + 1, 3 * i + 1) = -N22(i);
            B(5 * q + 2, 3 * i    ) = -T(2) * N12(i);
            B(5 * q + 2, 3 * i + 1) = -N12(i);
            B(5 * q + 2, 3 * i + 2) = -N12(i);

            // B_s = [ 0   N_{i|1}  0       ]   γ_1
            //       [ 0   0        N_{i|2} ]   γ_2
            B(5 * q + 3, 3 * i + 1) = basis.N_d1[0](q, i);
            B(5 * q + 4, 3 * i + 2) = basis.N_d1[1](q, i);
        }
    }
    return B;
}


template <std::floating_point T> Matrix<T>
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

template <std::floating_point T> Matrix<T>
PlateReissnerMindlinDispl3p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                          const BasisDerivs<T, 2>& basis,
                                                          const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * n);

    for (Index i = 0; i < n; ++i) 
    {
        // N_w = [ N_i  N_i  N_i ]
        Nw.col(3 * i    ) = basis.N.col(i);
        Nw.col(3 * i + 1) = basis.N.col(i);
        Nw.col(3 * i + 2) = basis.N.col(i);
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl3p<T>::rotation_shape_matrix(
    const Patch<T, 2>& /*patch*/, const BasisDerivs<T, 2>& basis, const IntrinsicGeometry<T, 2>& /*ig*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * n);

    for (Index q = 0; q < Q; ++q) 
    {
        for (Index i = 0; i < n; ++i) 
        {
            // N_rot = [ -N_{i|1}   0         -N_{i|1} ]
            //         [ -N_{i|2}  -N_{i|2}    0       ]
            Nphi(2 * q,     3 * i    ) = -basis.N_d1[0](q, i);
            Nphi(2 * q,     3 * i + 2) = -basis.N_d1[0](q, i);
            Nphi(2 * q + 1, 3 * i)     = -basis.N_d1[1](q, i);
            Nphi(2 * q + 1, 3 * i + 1) = -basis.N_d1[1](q, i);
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
