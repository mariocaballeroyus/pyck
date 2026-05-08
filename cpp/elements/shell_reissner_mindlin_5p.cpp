#include "shell_reissner_mindlin_5p.hpp"
#include "patch.hpp"

namespace pyck
{

template <std::floating_point T>
ShellReissnerMindlin5p<T>::ShellReissnerMindlin5p(Ptr<PlaneStressShell<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellReissnerMindlin5p: material is null.");
    }
}

template <std::floating_point T>
void ShellReissnerMindlin5p<T>::compute_local_stiffness(
    const Patch<T, 2>& patch,
    Index elem_idx,
    const ColMatrix<T, 2>& mapped_pts,
    const Vector<T>& q_weights,
    Matrix<T>& stiffness) const
{
    // ---- Surface geometry: tangents, director, inverse metric, jac.
    auto [a1, a2, a3, g_inv, b, jac] =
        patch.eval_surface_geometry(mapped_pts, elem_idx);

    // ---- Parametric basis derivatives (order 2: needed for a_{αβ} → Christoffels).
    auto spans        = patch.decode_span(elem_idx);
    auto basis_derivs = patch.tensor_product().eval_on_span(mapped_pts, spans, Index(2));
    auto act_pts      = patch.active_control_pts(spans);

    // For order=2: idx = du * 3 + dv.
    const Matrix<T>& N    = basis_derivs[0 * 3 + 0];
    const Matrix<T>& N_u  = basis_derivs[1 * 3 + 0];
    const Matrix<T>& N_v  = basis_derivs[0 * 3 + 1];
    const Matrix<T>& N_uu = basis_derivs[2 * 3 + 0];
    const Matrix<T>& N_uv = basis_derivs[1 * 3 + 1];
    const Matrix<T>& N_vv = basis_derivs[0 * 3 + 2];

    // Geometry second derivatives at each quadrature point.
    ColMatrix<T, 3> a11 = N_uu * act_pts;
    ColMatrix<T, 3> a12 = N_uv * act_pts;
    ColMatrix<T, 3> a22 = N_vv * act_pts;

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

        Eigen::Matrix<T, 3, 1> a1_q  = a1.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a2_q  = a2.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a3_q  = a3.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a11_q = a11.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a12_q = a12.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a22_q = a22.row(q).transpose();
        Eigen::Matrix<T, 3, 1> g_inv_q = g_inv.row(q).transpose();

        const T gi11 = g_inv_q(0);
        const T gi12 = g_inv_q(1);
        const T gi22 = g_inv_q(2);

        // Dual basis a^γ = g^{γσ} a_σ.
        Eigen::Matrix<T, 3, 1> aup1 = gi11 * a1_q + gi12 * a2_q;
        Eigen::Matrix<T, 3, 1> aup2 = gi12 * a1_q + gi22 * a2_q;

        // Christoffels Γ^δ_{αβ} = a^δ · a_{αβ}.
        const T G1_11 = aup1.dot(a11_q);
        const T G1_12 = aup1.dot(a12_q);
        const T G1_22 = aup1.dot(a22_q);
        const T G2_11 = aup2.dot(a11_q);
        const T G2_12 = aup2.dot(a12_q);
        const T G2_22 = aup2.dot(a22_q);

        // Per-node B-matrix contributions.
        for (Index i = 0; i < n; ++i)
        {
            const T Ni   = N  (q, i);
            const T Ni_u = N_u(q, i);
            const T Ni_v = N_v(q, i);

            // Membrane B_m: rows (ε_{11}, ε_{22}, 2ε_{12}); only displacement DOFs.
            for (Index k = 0; k < 3; ++k) {
                Bm(0, 5 * i + k) = Ni_u * a1_q(k);
                Bm(1, 5 * i + k) = Ni_v * a2_q(k);
                Bm(2, 5 * i + k) = Ni_u * a2_q(k) + Ni_v * a1_q(k);
            }

            // Bending B_b: rows (χ_{11}, χ_{22}, 2χ_{12}); only rotation DOFs.
            Bb(0, 5 * i + 3) =  Ni_u - G1_11 * Ni;
            Bb(0, 5 * i + 4) =       - G2_11 * Ni;
            Bb(1, 5 * i + 3) =       - G1_22 * Ni;
            Bb(1, 5 * i + 4) =  Ni_v - G2_22 * Ni;
            Bb(2, 5 * i + 3) =  Ni_v - T(2) * G1_12 * Ni;
            Bb(2, 5 * i + 4) =  Ni_u - T(2) * G2_12 * Ni;

            // Shear B_s: rows (γ_1, γ_2); displacements via a_3, plus θ_α.
            for (Index k = 0; k < 3; ++k) {
                Bs(0, 5 * i + k) = Ni_u * a3_q(k);
                Bs(1, 5 * i + k) = Ni_v * a3_q(k);
            }
            Bs(0, 5 * i + 3) = Ni;
            Bs(1, 5 * i + 4) = Ni;
        }

        // Constitutive matrices (vary across the patch on curved surfaces).
        const Matrix<T> Dm = material_->membrane_voigt(g_inv_q);
        const Matrix<T> Db = material_->bending_voigt (g_inv_q);
        const Matrix<T> Ds = material_->shear_voigt   (g_inv_q);

        const T dV = q_weights(q) * jac(q);
        stiffness.noalias() += dV * (Bm.transpose() * Dm * Bm
                                   + Bb.transpose() * Db * Bb
                                   + Bs.transpose() * Ds * Bs);
    }
}

// ---- Plate-style virtuals: stubs returning zero matrices of the right shape.
// These exist only so existing flat-plate boundary conditions don't crash if
// applied to a shell. They do NOT reflect the shell's true strain measures.

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const Index Q = shape_derivs[idx::val].rows();
    const Index n = shape_derivs[idx::val].cols();
    return Matrix<T>::Zero(3 * Q, 5 * n);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const Index Q = shape_derivs[idx::val].rows();
    const Index n = shape_derivs[idx::val].cols();
    return Matrix<T>::Zero(2 * Q, 5 * n);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::bending_constitutive_matrix() const
{
    return Matrix<T>::Zero(3, 3);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const Index Q = shape_derivs[idx::val].rows();
    const Index n = shape_derivs[idx::val].cols();
    return Matrix<T>::Zero(Q, 5 * n);
}

template <std::floating_point T>
Matrix<T> ShellReissnerMindlin5p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const Index Q = shape_derivs[idx::val].rows();
    const Index n = shape_derivs[idx::val].cols();
    return Matrix<T>::Zero(2 * Q, 5 * n);
}

// === Template Instantiations ========================================================

template class ShellReissnerMindlin5p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellReissnerMindlin5p<float>;
#endif

} // namespace pyck
