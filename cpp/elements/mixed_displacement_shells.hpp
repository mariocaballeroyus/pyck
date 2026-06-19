#ifndef PYCK_MIXED_DISPLACEMENT_SHELLS_HPP
#define PYCK_MIXED_DISPLACEMENT_SHELLS_HPP

#include <concepts>
#include <utility>
#include <Eigen/Dense>

#include "element.hpp"
#include "element_values.hpp"
#include "shell_reissner_mindlin_hier_4p.hpp"
#include "shell_reissner_mindlin_hier_5p.hpp"
#include "../multi_index.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Mixed-Displacement (MD) hierarchic shells and their shared kernels
 *        (Bieber, Oesterle, Ramm, Bischoff, IJNME 114 (2018) 801-827).
 *
 * @details Membrane locking is removed by a 3-component strain-displacement field
 *          \f$\tilde{\mathbf u}=[\tilde v_{(11)},\tilde v_{(12)},\tilde v_{(22)}]\f$,
 *          equal-order with the displacement, whose strain is a *derivative*
 *          \f$\mathbf E_{\tilde u}=\tilde{\mathcal L}\tilde{\mathbf u}
 *          =[\tilde v_{(11),1};\,\tilde v_{(12),12};\,\tilde v_{(22),2}]\f$. The element
 *          stiffness is the symmetric-indefinite mixed block (no condensation):
 *          \f[
 *            \mathbf K_e=\begin{bmatrix}-\mathbf H_e&\mathbf G_e\\
 *            \mathbf G_e^\top&\mathbf K_{bs}\end{bmatrix},\quad
 *            \mathbf H_e=\!\int\tilde{\mathbf B}^\top\mathbf C_m\tilde{\mathbf B},\;
 *            \mathbf G_e=\!\int\tilde{\mathbf B}^\top\mathbf C_m\mathbf B_m,\;
 *            \mathbf K_{bs}=\!\int(\mathbf B_b^\top\mathbf D_b\mathbf B_b
 *            +\mathbf B_s^\top\mathbf D_s\mathbf B_s),
 *          \f]
 *          with per-node DOFs ordered \f$[\,\text{displacement (ndof\_base)};\,
 *          \tilde{\mathbf u}\,(3)\,]\f$. The membrane is absent from the displacement
 *          block; it enters only through \f$\mathbf G_e\f$. The element classes are thin
 *          wrappers over the `md::` kernels below.
 */
namespace md
{

/// @brief Number of membrane (in-plane) strain components in 2D.
inline constexpr Index kMembrane = 3;

/**
 * @brief Build the MD membrane strain-displacement operator \f$\tilde{\mathbf B}=\tilde{\mathcal L}
 *        \mathbf N\f$ at quadrature point @p q into @p Bt (kMembrane × 3N). Following Bieber et
 *        al. Eq. (26), \f$\tilde{\mathcal L}\f$ is built from **plain coordinate partials** of
 *        the strain-displacements: row 0 (ε₁₁) ← ∂ṽ₁₁/∂ξ¹, row 1 (ε₂₂) ← ∂ṽ₂₂/∂ξ², row 2
 *        (2ε₁₂) ← ∂²ṽ₁₂/∂ξ¹∂ξ². No Christoffel/metric correction enters \f$\tilde{\mathcal L}\f$
 *        — it is the parametric mixed second derivative, not the covariant Hessian.
 */
template <std::floating_point T>
inline void mixed_strain_matrix(const ElementValues<T, 2>& ev, Index q, Index N, Matrix<T>& Bt)
{
    constexpr Index n_d2 = 3;                          // 2D second-derivative components
    const Index     mixed = pack2<2>(0, 1);            // packed index of ∂²/∂ξ¹∂ξ²
    Bt.setZero(kMembrane, kMembrane * N);
    const auto dN = ev.dN(q);
    for (Index i = 0; i < N; ++i) {
        Bt(0, kMembrane * i + 0) = dN(i, 0);                              // ε₁₁  ← ∂ṽ₁₁/∂ξ¹
        Bt(1, kMembrane * i + 2) = dN(i, 1);                              // ε₂₂  ← ∂ṽ₂₂/∂ξ²
        Bt(2, kMembrane * i + 1) = ev.basis_derivs[2](i * n_d2 + mixed, q); // 2ε₁₂ ← ∂²ṽ₁₂/∂ξ¹∂ξ²
    }
}

/**
 * @brief Assemble the local MD mixed stiffness into @p K_local (interleaved layout,
 *        (ndof_base+3)·N square). Reuses @p element 's `strain_matrix` (membrane rows
 *        for the coupling, bending+shear rows for the primal block) and
 *        `constitutive_matrix` (membrane / bending / shear blocks).
 */
template <std::floating_point T>
inline void assemble_local_stiffness(const Element<T, 2>& element, Index ndof_base,
                                     const ElementValues<T, 2>& ev, Matrix<T>& K_local)
{
    element.strain_matrix(ev);                 // fills element.B_voigt_ : (8Q × ndof_base·N)
    const Matrix<T>& B = element.B_voigt_;
    const Index Q  = ev.num_points();
    const Index N  = static_cast<Index>(ev.basis_derivs[0].rows());
    const Index ns = static_cast<Index>(B.rows()) / Q;   // 8: membrane(3)+bending(3)+shear(2)
    const Index nu = ndof_base * N;                       // displacement DOFs
    const Index nt = kMembrane * N;                       // ũ DOFs
    const Index nd = ndof_base + kMembrane;               // DOFs per node (MD)

    Matrix<T> Kbs = Matrix<T>::Zero(nu, nu);   // bending + shear (primal, displacement)
    Matrix<T> H   = Matrix<T>::Zero(nt, nt);   // ũ membrane compliance
    Matrix<T> G   = Matrix<T>::Zero(nt, nu);   // coupling: ũ rows × displacement cols
    Matrix<T> Bt(kMembrane, nt);

    for (Index q = 0; q < Q; ++q) {
        const T dV = ev.mapped_weights_(q) * ev.jac(q);
        const ConstitutiveMatrix<T>  D  = element.constitutive_matrix(ev, q);
        const Eigen::Matrix<T, 3, 3> Cm = D.template block<3, 3>(0, 0);
        const Eigen::Matrix<T, 3, 3> Db = D.template block<3, 3>(3, 3);
        const Eigen::Matrix<T, 2, 2> Ds = D.template block<2, 2>(6, 6);

        const auto Bm = B.middleRows(ns * q + 0, 3);   // membrane (3 × nu)
        const auto Bb = B.middleRows(ns * q + 3, 3);   // bending  (3 × nu)
        const auto Bs = B.middleRows(ns * q + 6, 2);   // shear    (2 × nu)

        Kbs.noalias() += dV * (Bb.transpose() * (Db * Bb));
        Kbs.noalias() += dV * (Bs.transpose() * (Ds * Bs));

        mixed_strain_matrix<T>(ev, q, N, Bt);
        H.noalias() += dV * (Bt.transpose() * (Cm * Bt));
        G.noalias() += dV * (Bt.transpose() * (Cm * Bm));
    }

    // The ũ block H is singular by construction (the integration-constant modes carry no
    // membrane strain); the caller removes those zero-energy modes with Dirichlet pins on
    // the boundary ũ DOFs (see `membrane_md_boundary_dofs`).

    // Scatter into the interleaved per-node layout [displacement (ndof_base); ũ (3)].
    K_local.setZero(nd * N, nd * N);
    for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
            // displacement–displacement: bending + shear
            for (Index a = 0; a < ndof_base; ++a)
                for (Index b = 0; b < ndof_base; ++b)
                    K_local(nd * i + a, nd * j + b) += Kbs(ndof_base * i + a, ndof_base * j + b);
            // ũ blocks
            for (Index c = 0; c < kMembrane; ++c) {
                for (Index e = 0; e < kMembrane; ++e)
                    K_local(nd * i + ndof_base + c, nd * j + ndof_base + e)
                        += -H(kMembrane * i + c, kMembrane * j + e);
                for (Index b = 0; b < ndof_base; ++b) {
                    const T g = G(kMembrane * i + c, ndof_base * j + b);
                    K_local(nd * i + ndof_base + c, nd * j + b) += g;   // ũ row, u col
                    K_local(nd * j + b, nd * i + ndof_base + c) += g;   // transpose
                }
            }
        }
    }
}

/**
 * @brief Widen a per-node shape/strain output @p M from the base layout
 *        (R × ndof_base·N) to the MD layout (R × (ndof_base+3)·N), placing the base
 *        columns in the displacement slots and leaving the ũ slots zero. In place.
 */
template <std::floating_point T>
inline void widen_dof_columns(Matrix<T>& M, Index ndof_base, Index N)
{
    const Index nd = ndof_base + kMembrane;
    const Matrix<T> src = M;                     // copy: (R × ndof_base·N)
    M.setZero(src.rows(), nd * N);
    for (Index i = 0; i < N; ++i)
        M.middleCols(nd * i, ndof_base) = src.middleCols(ndof_base * i, ndof_base);
}

/**
 * @brief Build the MD generalised-stress shape matrix into @p N_sigma (8Q × (ndof_base+3)·N).
 *        The **membrane** stress is recovered from the ũ field (n = C_m·B̃·ũ, in the ũ slots);
 *        the **bending** and **transverse-shear** stresses stay primal (D_b·B_b, D_s·B_s, in
 *        the displacement slots). Drives the natural (traction / moment) boundary traces and
 *        the recovered force/moment resultants.
 */
template <std::floating_point T>
inline void assemble_stress_shape(const Element<T, 2>& element, Index ndof_base,
                                  const ElementValues<T, 2>& ev, Matrix<T>& N_sigma)
{
    element.strain_matrix(ev);                 // base B (8Q × ndof_base·N) for bending/shear
    const Matrix<T>& B = element.B_voigt_;
    const Index Q  = ev.num_points();
    const Index N  = static_cast<Index>(ev.basis_derivs[0].rows());
    const Index ns = static_cast<Index>(B.rows()) / Q;   // 8
    const Index nd = ndof_base + kMembrane;

    N_sigma.setZero(ns * Q, nd * N);
    Matrix<T> Bt(kMembrane, kMembrane * N);
    for (Index q = 0; q < Q; ++q) {
        const ConstitutiveMatrix<T>  D  = element.constitutive_matrix(ev, q);
        const Eigen::Matrix<T, 3, 3> Cm = D.template block<3, 3>(0, 0);
        const Eigen::Matrix<T, 3, 3> Db = D.template block<3, 3>(3, 3);
        const Eigen::Matrix<T, 2, 2> Ds = D.template block<2, 2>(6, 6);

        mixed_strain_matrix<T>(ev, q, N, Bt);
        const Matrix<T> Sm = Cm * Bt;                              // membrane stress ← ũ
        const Matrix<T> Sb = Db * B.middleRows(ns * q + 3, 3);     // bending  ← displacement
        const Matrix<T> Ss = Ds * B.middleRows(ns * q + 6, 2);     // shear    ← displacement

        for (Index i = 0; i < N; ++i) {
            for (Index r = 0; r < kMembrane; ++r)
                for (Index c = 0; c < kMembrane; ++c)
                    N_sigma(ns * q + r, nd * i + ndof_base + c) = Sm(r, kMembrane * i + c);
            for (Index r = 0; r < 3; ++r)
                for (Index b = 0; b < ndof_base; ++b)
                    N_sigma(ns * q + 3 + r, nd * i + b) = Sb(r, ndof_base * i + b);
            for (Index r = 0; r < 2; ++r)
                for (Index b = 0; b < ndof_base; ++b)
                    N_sigma(ns * q + 6 + r, nd * i + b) = Ss(r, ndof_base * i + b);
        }
    }
}

} // namespace md

/**
 * @brief Mixed-Displacement (MD) hierarchic four-parameter Reissner-Mindlin shell.
 *
 * @details Same kinematics as ShellReissnerMindlinHier4p (rotation-free), but membrane
 *          locking is removed by the Mixed-Displacement method: a 3-component, equal-order
 *          membrane strain-displacement field (slots 4..6) supplies the lower-order membrane
 *          strain through a symmetric-indefinite mixed stiffness (no condensation); bending
 *          and transverse shear stay primal.
 *
 *              slot 0..2 : Cartesian displacements (v_x, v_y, v_z)
 *              slot 3    : twist potential psi
 *              slot 4..6 : membrane strain-displacements (ṽ_11, ṽ_12, ṽ_22)
 *
 * @note The \f$\tilde{\mathbf u}\f$ field has integration-constant zero-energy modes that must
 *       be stabilised by Dirichlet boundary conditions on its boundary DOFs.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlinHier4pMD : public ShellReissnerMindlinHier4p<T>
{
public:

    explicit ShellReissnerMindlinHier4pMD(Ptr<PlaneStress2d<T>> material)
        : ShellReissnerMindlinHier4p<T>(std::move(material)) {}

    /// @brief Displacement (4) + membrane strain-displacements (3).
    std::size_t num_node_dofs() const override { return 7; }

    /// @brief Local mixed stiffness [[-H, G],[Gᵀ, K_bending+shear]] over the 7-DOF layout.
    void compute_local_stiffness(const ElementValues<T, 2>& ev, Matrix<T>& stiffness) const override
    {
        md::assemble_local_stiffness<T>(*this, /*ndof_base=*/4, ev, stiffness);
    }

    /// @brief Displacement shape on the 7-DOF layout (displacement slots only; ũ slots zero).
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        ShellReissnerMindlinHier4p<T>::displacement_shape_matrix(ev);
        md::widen_dof_columns<T>(this->N_w_, /*ndof_base=*/4,
                                 static_cast<Index>(ev.basis_derivs[0].rows()));
    }

    /// @brief Director-tilt shape on the 7-DOF layout (for weak rotation BCs).
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        ShellReissnerMindlinHier4p<T>::rotation_shape_matrix(ev);
        md::widen_dof_columns<T>(this->N_phi_, /*ndof_base=*/4,
                                 static_cast<Index>(ev.basis_derivs[0].rows()));
    }

    /// @brief Generalised-stress shape on the 7-DOF layout: membrane stress from ũ,
    ///        bending/shear from displacement (for traction/moment BCs and recovery).
    void stress_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        md::assemble_stress_shape<T>(*this, /*ndof_base=*/4, ev, this->N_sigma_);
    }
};

/**
 * @brief Mixed-Displacement (MD) hierarchic five-parameter Reissner-Mindlin shell.
 *
 * @details Same kinematics as ShellReissnerMindlinHier5p, but membrane locking is removed by
 *          the Mixed-Displacement method: a 3-component, equal-order membrane
 *          strain-displacement field (slots 5..7) supplies the lower-order membrane strain
 *          through a symmetric-indefinite mixed stiffness (no condensation); bending and
 *          transverse shear stay primal.
 *
 *              slot 0..2 : Cartesian displacements (v_x, v_y, v_z)
 *              slot 3..4 : hierarchic difference vector (w^1, w^2)
 *              slot 5..7 : membrane strain-displacements (ṽ_11, ṽ_12, ṽ_22)
 *
 * @note The \f$\tilde{\mathbf u}\f$ field has integration-constant zero-energy modes that must
 *       be stabilised by Dirichlet boundary conditions on its boundary DOFs.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class ShellReissnerMindlinHier5pMD : public ShellReissnerMindlinHier5p<T>
{
public:

    explicit ShellReissnerMindlinHier5pMD(Ptr<PlaneStress2d<T>> material)
        : ShellReissnerMindlinHier5p<T>(std::move(material)) {}

    /// @brief Displacement (5) + membrane strain-displacements (3).
    std::size_t num_node_dofs() const override { return 8; }

    /// @brief Local mixed stiffness [[-H, G],[Gᵀ, K_bending+shear]] over the 8-DOF layout.
    void compute_local_stiffness(const ElementValues<T, 2>& ev, Matrix<T>& stiffness) const override
    {
        md::assemble_local_stiffness<T>(*this, /*ndof_base=*/5, ev, stiffness);
    }

    /// @brief Displacement shape on the 8-DOF layout (displacement slots only; ũ slots zero).
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        ShellReissnerMindlinHier5p<T>::displacement_shape_matrix(ev);
        md::widen_dof_columns<T>(this->N_w_, /*ndof_base=*/5,
                                 static_cast<Index>(ev.basis_derivs[0].rows()));
    }

    /// @brief Director-tilt shape on the 8-DOF layout (for weak rotation BCs).
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        ShellReissnerMindlinHier5p<T>::rotation_shape_matrix(ev);
        md::widen_dof_columns<T>(this->N_phi_, /*ndof_base=*/5,
                                 static_cast<Index>(ev.basis_derivs[0].rows()));
    }

    /// @brief Generalised-stress shape on the 8-DOF layout: membrane stress from ũ,
    ///        bending/shear from displacement (for traction/moment BCs and recovery).
    void stress_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        md::assemble_stress_shape<T>(*this, /*ndof_base=*/5, ev, this->N_sigma_);
    }
};

} // namespace pyck

#endif // PYCK_MIXED_DISPLACEMENT_SHELLS_HPP
