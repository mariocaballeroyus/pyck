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

        const auto a1_q = ev.a(0).row(q);
        const auto a2_q = ev.a(1).row(q);
        const auto a3_q = a_3.row(q);

        // Reference-normal derivatives A_{3,β}, for the displacement-curvature
        // coupling in the bending block.
        const auto A3_d1_q = ev.n_d1(0).row(q);
        const auto A3_d2_q = ev.n_d1(1).row(q);

        const T Gam1_11 = ev.Gamma(0, 0, 0)(q);
        const T Gam1_12 = ev.Gamma(0, 0, 1)(q);
        const T Gam1_22 = ev.Gamma(0, 1, 1)(q);
        const T Gam2_11 = ev.Gamma(1, 0, 0)(q);
        const T Gam2_12 = ev.Gamma(1, 0, 1)(q);
        const T Gam2_22 = ev.Gamma(1, 1, 1)(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i   = slab0(i);
            const T N_u_i = slab1(i * 2 + 0);
            const T N_v_i = slab1(i * 2 + 1);

            // Membrane: B_m
            for (Index k = 0; k < 3; ++k) {
                B(8*q,     5*i + k) = N_u_i * a1_q(k);
                B(8*q + 1, 5*i + k) = N_v_i * a2_q(k);
                B(8*q + 2, 5*i + k) = N_u_i * a2_q(k) + N_v_i * a1_q(k);
            }

            // Bending and twisting: B_b
            // Rotation-derivative contribution: A_α · w_,β  =  φ_α|β.
            B(8*q + 3, 5*i + 3) =  N_u_i - Gam1_11 * N_i;
            B(8*q + 3, 5*i + 4) =        - Gam2_11 * N_i;
            B(8*q + 4, 5*i + 3) =        - Gam1_22 * N_i;
            B(8*q + 4, 5*i + 4) =  N_v_i - Gam2_22 * N_i;
            B(8*q + 5, 5*i + 3) =  N_v_i - T(2) * Gam1_12 * N_i;
            B(8*q + 5, 5*i + 4) =  N_u_i - T(2) * Gam2_12 * N_i;
            // Displacement-curvature coupling: u_,α · A_{3,β}. Vanishes for a
            // flat reference (A_{3,β} = 0); essential for curved shells.
            for (Index k = 0; k < 3; ++k) {
                B(8*q + 3, 5*i + k) = N_u_i * A3_d1_q(k);
                B(8*q + 4, 5*i + k) = N_v_i * A3_d2_q(k);
                B(8*q + 5, 5*i + k) = N_u_i * A3_d2_q(k) + N_v_i * A3_d1_q(k);
            }

            // Transverse shear: B_s
            for (Index k = 0; k < 3; ++k) {
                B(8*q + 6, 5*i + k) = N_u_i * a3_q(k);
                B(8*q + 7, 5*i + k) = N_v_i * a3_q(k);
            }
            B(8*q + 6, 5*i + 3) = N_i;
            B(8*q + 7, 5*i + 4) = N_i;
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
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    this->N_w_workspace_.setZero(Q, 5 * N);
    // TODO: implement this
}

template <std::floating_point T>
void
ShellReissnerMindlin5p<T>::rotation_shape_matrix(const ElementValues<T, 2>& ev) const
{
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    this->N_phi_workspace_.setZero(2 * Q, 5 * N);
    // TODO: implement this
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin5p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin5p<float>;
#endif

} // namespace pyck
