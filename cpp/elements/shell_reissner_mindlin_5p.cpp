#include "shell_reissner_mindlin_5p.hpp"
#include "patch.hpp"
#include "basis_derivs.hpp"
#include "local_frame.hpp"
#include "christoffels.hpp"
#include "directors.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlin5p<T>::ShellReissnerMindlin5p(Ptr<PlaneStressShell<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlin5p: "
                                    "material is null.");
    }
}

template <std::floating_point T> void 
ShellReissnerMindlin5p<T>::compute_local_stiffness(const Patch<T, 2>& patch, 
                                                   Index elem_idx, 
                                                   const ColMatrix<T, 2>& mapped_pts,
                                                   const Vector<T>& q_weights, 
                                                   Matrix<T>& stiffness) const
{
    auto basis        = eval_basis(patch, mapped_pts, elem_idx, 2);
    auto act_pts      = patch.active_control_pts(elem_idx);
    auto local        = eval_local_frame(basis, act_pts);
    auto christoffel  = eval_christoffel(local);
    auto a_3          = eval_normal(local);

    const Matrix<T>& N    = basis.N;
    const Matrix<T>& N_u  = basis.N_u;
    const Matrix<T>& N_v  = basis.N_v;

    const Index Q     = mapped_pts.rows();
    const Index n     = N.cols();
    const Index K_dof = 5 * n;

    stiffness.setZero(K_dof, K_dof);

    Matrix<T> Bm(3, K_dof);
    Matrix<T> Bb(3, K_dof);
    Matrix<T> Bs(2, K_dof);

    for (Index q = 0; q < Q; ++q)
    {
        Bm.setZero();
        Bb.setZero();
        Bs.setZero();

        const auto a1_q = local.a1.row(q);
        const auto a2_q = local.a2.row(q);
        const auto a3_q = a_3.row(q);
        Eigen::Matrix<T, 3, 1> g_inv_q = local.g_inv.row(q).transpose();

        const T Gam1_11 = christoffel.G1_11(q);
        const T Gam1_12 = christoffel.G1_12(q);
        const T Gam1_22 = christoffel.G1_22(q);
        const T Gam2_11 = christoffel.G2_11(q);
        const T Gam2_12 = christoffel.G2_12(q);
        const T Gam2_22 = christoffel.G2_22(q);

        for (Index i = 0; i < n; ++i)
        {
            const T Ni   = N  (q, i);
            const T Ni_u = N_u(q, i);
            const T Ni_v = N_v(q, i);

            // B_m = [ N_{i|1} (a_1)_x   N_{i|1} (a_1)_y   N_{i|1} (a_1)_z   0   0 ]
            //       [ N_{i|2} (a_2)_x   N_{i|2} (a_2)_y   N_{i|2} (a_2)_z   0   0 ]
            //       [ N_{i|1} (a_2)_x + N_{i|2} (a_1)_x   ...               0   0 ]
            for (Index k = 0; k < 3; ++k) 
            {
                Bm(0, 5 * i + k) = Ni_u * a1_q(k);
                Bm(1, 5 * i + k) = Ni_v * a2_q(k);
                Bm(2, 5 * i + k) = Ni_u * a2_q(k) + Ni_v * a1_q(k);
            }

            // B_b = [ 0  0  0   N_{i|1} − Γ¹_{11} N_i      −Γ²_{11} N_i            ]
            //       [ 0  0  0   −Γ¹_{22} N_i               N_{i|2} − Γ²_{22} N_i   ]
            //       [ 0  0  0   N_{i|2} − 2 Γ¹_{12} N_i    N_{i|1} − 2 Γ²_{12} N_i ]
            Bb(0, 5 * i + 3) =  Ni_u - Gam1_11 * Ni;
            Bb(0, 5 * i + 4) =       - Gam2_11 * Ni;
            Bb(1, 5 * i + 3) =       - Gam1_22 * Ni;
            Bb(1, 5 * i + 4) =  Ni_v - Gam2_22 * Ni;
            Bb(2, 5 * i + 3) =  Ni_v - T(2) * Gam1_12 * Ni;
            Bb(2, 5 * i + 4) =  Ni_u - T(2) * Gam2_12 * Ni;

            // B_s = [ N_{i|1} (a_3)_x   N_{i|1} (a_3)_y   N_{i|1} (a_3)_z   N_i  0  ]
            //       [ N_{i|2} (a_3)_x   N_{i|2} (a_3)_y   N_{i|2} (a_3)_z   0   N_i ]
            for (Index k = 0; k < 3; ++k) 
            {
                Bs(0, 5 * i + k) = Ni_u * a3_q(k);
                Bs(1, 5 * i + k) = Ni_v * a3_q(k);
            }
            Bs(0, 5 * i + 3) = Ni;
            Bs(1, 5 * i + 4) = Ni;
        }

        // D_m = t·H, D_b = (t³/12)·H, D_s = shear_voigt; all per-qp.
        const Eigen::Matrix<T, 3, 3> H  = material_->surface_H_voigt(g_inv_q);
        const Eigen::Matrix<T, 3, 3> Dm = material_->thickness() * H;
        const Eigen::Matrix<T, 3, 3> Db = (material_->thickness() *
                                           material_->thickness() *
                                           material_->thickness() / T(12)) * H;
        const Eigen::Matrix<T, 2, 2> Ds = material_->shear_voigt(g_inv_q);

        const T dV = q_weights(q) * local.jac(q);
        stiffness.noalias() += dV * (Bm.transpose() * Dm * Bm
                                   + Bb.transpose() * Db * Bb
                                   + Bs.transpose() * Ds * Bs);
    }
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::bending_strain_matrix(
    const Patch<T, 2>& /*patch*/, const BasisDerivs<T, 2>& basis, const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    return Matrix<T>::Zero(3 * Q, 5 * n);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::shear_strain_matrix(
    const Patch<T, 2>& /*patch*/, const BasisDerivs<T, 2>& basis, const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    return Matrix<T>::Zero(2 * Q, 5 * n);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::bending_constitutive_matrix(
    const LocalFrame<T, 2>& /*local*/, Index /*q*/) const
{
    return Matrix<T>::Zero(3, 3);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::displacement_shape_matrix(
    const Patch<T, 2>& /*patch*/, const BasisDerivs<T, 2>& basis, const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    return Matrix<T>::Zero(Q, 5 * n);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::rotation_shape_matrix(
    const Patch<T, 2>& /*patch*/, const BasisDerivs<T, 2>& basis, const LocalFrame<T, 2>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    return Matrix<T>::Zero(2 * Q, 5 * n);
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin5p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin5p<float>;
#endif

} // namespace pyck
