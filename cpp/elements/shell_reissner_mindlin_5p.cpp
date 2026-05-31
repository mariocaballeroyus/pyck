#include "shell_reissner_mindlin_5p.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "surface_geometry.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlin5p<T>::ShellReissnerMindlin5p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlin5p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    const ColMatrix<T, 3>& a_3 = ev.n;

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(8 * Q, 5 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = ev.results_[0].col(q);
        auto slab1 = ev.results_[1].col(q);

        const auto A1   = ev.a(0).row(q);   // covariant tangent A_1
        const auto A2   = ev.a(1).row(q);   // covariant tangent A_2
        const auto A3 = a_3.row(q);         // unit normal A_3

        // Reference-normal derivatives A_{3,β}
        const auto A3_d1 = ev.n_d1(0).row(q);
        const auto A3_d2 = ev.n_d1(1).row(q);

        // Second surface derivatives A_{λ,β}
        const auto A1_d1 = ev.a_d1(0, 0).row(q);   // A_{1,1}
        const auto A1_d2 = ev.a_d1(0, 1).row(q);   // A_{1,2} = A_{2,1}
        const auto A2_d2 = ev.a_d1(1, 1).row(q);   // A_{2,2}

        // Metric A_α·A_β
        const T A11 = A1.dot(A1);   // A_{11}
        const T A12 = A1.dot(A2);   // A_{12} = A_{21}
        const T A22 = A2.dot(A2);   // A_{22}

        // Christoffels of the first kind Cα_λβ = A_α·A_{λ,β}
        const T C1_11 = A1.dot(A1_d1), C1_12 = A1.dot(A1_d2), C1_22 = A1.dot(A2_d2);
        const T C2_11 = A2.dot(A1_d1), C2_12 = A2.dot(A1_d2), C2_22 = A2.dot(A2_d2);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i   = slab0(i);
            const T N_u_i = slab1(i * 2 + 0);
            const T N_v_i = slab1(i * 2 + 1);

            // --- Membrane -----------------------------------------------------------

            for (Index k = 0; k < 3; ++k) {
                // \varepsilon_{11} = u_{i,1} · A_1
                B(8*q,     5*i + k) = N_u_i * A1(k);
                // \varepsilon_{22}  = u_{i,2} · A_2
                B(8*q + 1, 5*i + k) = N_v_i * A2(k);
                // 2\varepsilon_{12} = u_{i,1} · A_2 + u_{i,2} · A_1
                B(8*q + 2, 5*i + k) = N_u_i * A2(k) + N_v_i * A1(k);
            }

            // --- Bending ------------------------------------------------------------

            // Rotations (contravariant)

            // \kappa_{11} = \phi^λ_{,1} * (A_1·A_λ) + \phi^λ * (A_1·A_{λ,1})
            B(8*q + 3, 5*i + 3) = N_u_i * A11 + N_i * C1_11;
            B(8*q + 3, 5*i + 4) = N_u_i * A12 + N_i * C1_12;
            // \kappa_{22} = \phi^λ_{,2} * (A_2·A_λ) + \phi^λ * (A_2·A_{λ,2})
            B(8*q + 4, 5*i + 3) = N_v_i * A12 + N_i * C2_12;
            B(8*q + 4, 5*i + 4) = N_v_i * A22 + N_i * C2_22;
            // 2 \kappa_{12} = \phi^λ_{,2}(A_1·A_λ) + \phi^λ_{,1}(A_2·A_λ)
            //              + \phi^λ (A_1·A_{λ,2} + A_2·A_{λ,1})
            B(8*q + 5, 5*i + 3) = N_v_i * A11 + N_u_i * A12 + N_i * (C1_12 + C2_11);
            B(8*q + 5, 5*i + 4) = N_v_i * A12 + N_u_i * A22 + N_i * (C1_22 + C2_12);

            // Displacements (cartesian)

            for (Index k = 0; k < 3; ++k) {
                // \kappa_{11} += u_{k,1} (A_{3,1})_k
                B(8*q + 3, 5*i + k) = N_u_i * A3_d1(k);
                // \kappa_{22} += u_{k,2} (A_{3,2})_k
                B(8*q + 4, 5*i + k) = N_v_i * A3_d2(k);
                // 2 \kappa_{12} += u_{k,1}(A_{3,2})_k + u_{k,2}(A_{3,1})_k
                B(8*q + 5, 5*i + k) = N_u_i * A3_d2(k) + N_v_i * A3_d1(k);
            }

            // --- Transverse shear ---------------------------------------------------

            // Rotations (contravariant)
 
            // 2 \varepsilon_{13} = \phi^1 A_11 + \phi^2 A_21
            B(8*q + 6, 5*i + 3) = N_i * A11;
            B(8*q + 6, 5*i + 4) = N_i * A12;
            // 2 \varepsilon_{23} = \phi^1 A_12 + \phi^2 A_22
            B(8*q + 7, 5*i + 3) = N_i * A12;
            B(8*q + 7, 5*i + 4) = N_i * A22;

            // Displacements (cartesian)

            for (Index k = 0; k < 3; ++k) {
                // 2ε_13 = u_{k,1} (A_3)_k
                B(8*q + 6, 5*i + k) = N_u_i * A3(k);
                // 2ε_23 = u_{k,2} (A_3)_k
                B(8*q + 7, 5*i + k) = N_v_i * A3(k);
            }
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellReissnerMindlin5p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    // D = [ D_m  0    0   ]
    //     [ 0    D_b  0   ]
    //     [ 0    0    D_s ]
    const Eigen::Matrix<T, 3, 1> g_inv_q = g_inv_voigt(ev, q);
    const Eigen::Matrix<T, 3, 3> C  = material_->elasticity_voigt(g_inv_q);
    const T t = material_->thickness();
    const Eigen::Matrix<T, 3, 3> Dm = t * C;
    const Eigen::Matrix<T, 3, 3> Db = (t * t * t / T(12)) * C;
    const Eigen::Matrix<T, 2, 2> Ds = material_->shear_voigt(g_inv_q);

    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(8, 8);
    D.template block<3, 3>(0, 0) = Dm;
    D.template block<3, 3>(3, 3) = Db;
    D.template block<2, 2>(6, 6) = Ds;
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Physical displacement u = (u_x, u_y, u_z): the three Cartesian DOFs.
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Matrix<T>& U = this->N_w_workspace_;
    U.setZero(3 * Q, 5 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.results_[0].col(q);
        for (Index i = 0; i < N; ++i) {
            const T Ni = slab0(i);
            U(3 * q + 0, 5 * i + 0) = Ni;   // u_x
            U(3 * q + 1, 5 * i + 1) = Ni;   // u_y
            U(3 * q + 2, 5 * i + 2) = Ni;   // u_z
        }
    }
}

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::rotation_shape_matrix(const ElementValues<T, 2>&) const
{
    throw std::runtime_error("ShellReissnerMindlin5p::rotation_shape_matrix: "
                             "not implemented.");
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin5p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin5p<float>;
#endif

} // namespace pyck
