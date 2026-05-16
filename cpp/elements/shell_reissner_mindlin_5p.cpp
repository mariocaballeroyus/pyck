#include "shell_reissner_mindlin_5p.hpp"
#include "patch.hpp"
#include "basis_derivs.hpp"
#include "local_frame.hpp"
#include "christoffels.hpp"
#include "directors.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
ShellReissnerMindlin5p<T>::ShellReissnerMindlin5p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlin5p: "
                                    "material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
Matrix<T>
ShellReissnerMindlin5p<T>::strain_matrix(const Patch<T, 2>& /*patch*/,
                                         const BasisDerivs<T, 2>& basis,
                                         const LocalFrame<T, 2>& local,
                                         const ChristoffelSymbols<T, 2>& chr) const
{
    auto a_3  = eval_normal(local);

    const Matrix<T>& N    = basis.N;
    const Matrix<T>& N_u  = basis.N_u;
    const Matrix<T>& N_v  = basis.N_v;

    const Index Q = N.rows();
    const Index n = N.cols();
    Matrix<T> B = Matrix<T>::Zero(8 * Q, 5 * n);

    for (Index q = 0; q < Q; ++q)
    {
        const auto a1_q = local.a1.row(q);
        const auto a2_q = local.a2.row(q);
        const auto a3_q = a_3.row(q);

        const T Gam1_11 = chr.G1_11(q);
        const T Gam1_12 = chr.G1_12(q);
        const T Gam1_22 = chr.G1_22(q);
        const T Gam2_11 = chr.G2_11(q);
        const T Gam2_12 = chr.G2_12(q);
        const T Gam2_22 = chr.G2_22(q);

        for (Index i = 0; i < n; ++i)
        {
            const T Ni   = N  (q, i);
            const T Ni_u = N_u(q, i);
            const T Ni_v = N_v(q, i);

            // Membrane
            // B_m = [ N_{i|1} (a_1)_x   N_{i|1} (a_1)_y   N_{i|1} (a_1)_z   0   0 ]
            //       [ N_{i|2} (a_2)_x   N_{i|2} (a_2)_y   N_{i|2} (a_2)_z   0   0 ]
            //       [ N_{i|1} (a_2) + N_{i|2} (a_1)                         0   0 ]
            for (Index k = 0; k < 3; ++k)
            {
                B(8 * q    , 5 * i + k) = Ni_u * a1_q(k);
                B(8 * q + 1, 5 * i + k) = Ni_v * a2_q(k);
                B(8 * q + 2, 5 * i + k) = Ni_u * a2_q(k) + Ni_v * a1_q(k);
            }

            // Bending and twisting
            // B_b = [ 0  0  0   N_{i|1} − Γ¹_{11} N_i      −Γ²_{11} N_i            ]
            //       [ 0  0  0   −Γ¹_{22} N_i               N_{i|2} − Γ²_{22} N_i   ]
            //       [ 0  0  0   N_{i|2} − 2 Γ¹_{12} N_i    N_{i|1} − 2 Γ²_{12} N_i ]
            B(8 * q + 3, 5 * i + 3) =  Ni_u - Gam1_11 * Ni;
            B(8 * q + 3, 5 * i + 4) =       - Gam2_11 * Ni;
            B(8 * q + 4, 5 * i + 3) =       - Gam1_22 * Ni;
            B(8 * q + 4, 5 * i + 4) =  Ni_v - Gam2_22 * Ni;
            B(8 * q + 5, 5 * i + 3) =  Ni_v - T(2) * Gam1_12 * Ni;
            B(8 * q + 5, 5 * i + 4) =  Ni_u - T(2) * Gam2_12 * Ni;

            // Transverse shear
            // B_s = [ N_{i|1} (a_3)_x   N_{i|1} (a_3)_y   N_{i|1} (a_3)_z   N_i  0  ]
            //       [ N_{i|2} (a_3)_x   N_{i|2} (a_3)_y   N_{i|2} (a_3)_z   0   N_i ]
            for (Index k = 0; k < 3; ++k)
            {
                B(8 * q + 6, 5 * i + k) = Ni_u * a3_q(k);
                B(8 * q + 7, 5 * i + k) = Ni_v * a3_q(k);
            }
            B(8 * q + 6, 5 * i + 3) = Ni;
            B(8 * q + 7, 5 * i + 4) = Ni;
        }
    }
    return B;
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::constitutive_matrix(
    const LocalFrame<T, 2>& local, Index q) const
{
    // D = [ D_m  0    0   ]
    //     [ 0    D_b  0   ]
    //     [ 0    0    D_s ]
    const Eigen::Matrix<T, 3, 1> g_inv_q = local.g_inv.row(q).transpose();
    const Eigen::Matrix<T, 3, 3> C  = material_->surface_C_voigt(g_inv_q);
    const T t  = material_->thickness();
    const Eigen::Matrix<T, 3, 3> Dm = t * C;
    const Eigen::Matrix<T, 3, 3> Db = (t * t * t / T(12)) * C;
    const Eigen::Matrix<T, 2, 2> Ds = material_->shear_voigt(g_inv_q);

    Matrix<T> D = Matrix<T>::Zero(8, 8);
    D.template block<3, 3>(0, 0) = Dm;
    D.template block<3, 3>(3, 3) = Db;
    D.template block<2, 2>(6, 6) = Ds;
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
ShellReissnerMindlin5p<T>::displacement_shape_matrix(const Patch<T, 2>& /*patch*/,
                                                     const BasisDerivs<T, 2>& basis,
                                                     const LocalFrame<T, 2>& /*local*/,
                                                     const ChristoffelSymbols<T, 2>& /*chr*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> N_w = Matrix<T>::Zero(Q, 5 * n);
    Matrix<T> N_psi = Matrix<T>::Zero(2 * Q, 5 * n);

    // TODO: implement this
    return N_w;
}

template <std::floating_point T>
Matrix<T> 
ShellReissnerMindlin5p<T>::rotation_shape_matrix(const Patch<T, 2>& /*patch*/, 
                                                 const BasisDerivs<T, 2>& basis, 
                                                 const LocalFrame<T, 2>& /*local*/,
                                                 const ChristoffelSymbols<T, 2>& /*chr*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> N_psi = Matrix<T>::Zero(2 * Q, 5 * n);

    // TODO: implement this
    return N_psi;
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin5p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin5p<float>;
#endif

} // namespace pyck
