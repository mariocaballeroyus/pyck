#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <stdexcept>
#include <Eigen/Dense>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "element_values.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Field types supported by `Element::eval_field` and `integrate_on_patch`.
 *
 * Each tag selects which physical quantity is evaluated at the quadrature
 * points from `(basis, intrinsic geometry, u_local)`. Component count `k`
 * depends on the formulation; see the concrete element header.
 */
enum class FieldType
{
    DISPLACEMENT,  ///< Generalised displacement(s) — N_w · u_local.
    ROTATION,      ///< Generalised rotation(s)     — N_φ · u_local.
    STRAIN,        ///< Generalised strain          — B · u_local.
    STRESS,        ///< Generalised stress          — D · B · u_local.
};

/**
 * @brief Base class for structural elements of parametric dimension d.
 *
 * @tparam T Scalar type.
 * @tparam d Parametric dimension (1 = curve, 2 = plate/shell).
 */
template <std::floating_point T, std::size_t d>
class Element
{
public:
    virtual ~Element() = default;

    // === DOF layout =================================================================

    /// @brief Number of DOFs per node.
    virtual std::size_t num_node_dofs() const = 0;

    /// @brief Basis derivative order needed by this element's strain / stress
    ///        / shape matrix bodies.
    virtual Index basis_order() const = 0;

    /// @brief Quantity flag bitmask declaring which geometric quantities this
    ///        element's `strain_matrix` / `constitutive_matrix` /
    ///        `compute_local_stiffness` read. The bulk assembler uses this to
    ///        size the interior `ElementValues` workspace.
    virtual unsigned flags() const = 0;

    /// @brief Flags this element's *essential*-boundary shape matrices read
    ///        (`displacement_shape_matrix`, `rotation_shape_matrix`). Used by
    ///        kinematic boundary fields — those that constrain primary
    ///        unknowns (displacements, rotations).
    virtual unsigned essential_flags() const = 0;

    /// @brief Flags this element's `stress_matrix` reads (= what its
    ///        `strain_matrix` + `constitutive_matrix` consume). Used by
    ///        *natural*-boundary fields — those applying work-conjugate
    ///        tractions (shear, bending moment, twisting moment).
    virtual unsigned natural_flags() const = 0;

    // === Matrix Operators (Element Formulation-Agnostic) ============================

    /**
     * @brief Generalised-stress shape matrix S = D B. Writes into this
     *        element's own @ref N_sigma_workspace_; read the result via
     *        `element.N_sigma_workspace_` after the call. Internally
     *        reuses @ref B_workspace_ as scratch for B.
     */
    virtual void stress_matrix(const ElementValues<T, d>& ev) const;

    /**
     * @brief Local stiffness matrix K = ∫_Ω B^T D B dV.
     *
     * @param ev Per-element workspace bound via `ElementValues::reinit`.
     * @param stiffness The local stiffness matrix.
     */
    virtual void compute_local_stiffness(const ElementValues<T, d>& ev,
                                         Matrix<T>& stiffness) const;

    // === Matrix Operators (Element Formulation-Specific) ============================

    /**
     * @brief Strain-displacement operator B, row-stacked per qp. Writes
     *        into this element's own @ref B_workspace_.
     */
    virtual void strain_matrix(const ElementValues<T, d>& ev) const = 0;

    /**
     * @brief Constitutive operator D at quadrature point q.
     */
    virtual ConstitutiveMatrix<T>
    constitutive_matrix(const ElementValues<T, d>& ev, Index q) const = 0;

    // === Shape Matrices (Pure Virtual) ==============================================

    /**
     * @brief Transverse-displacement shape matrix N_w (Q × K). Writes into
     *        @ref N_w_workspace_.
     */
    virtual void displacement_shape_matrix(const ElementValues<T, d>& ev) const = 0;

    /**
     * @brief Rotation shape matrix N_φ. Writes into @ref N_phi_workspace_.
     */
    virtual void rotation_shape_matrix(const ElementValues<T, d>& ev) const = 0;

    // === Preallocated scratch =======================================================
    //
    // Per-formulation workspaces for the per-element shape matrices. Public
    // because in-process callers (boundary fields, LoadCondition, the
    // assembler's hot loop) write into them through a `const Element&` to
    // get heap-free per-call behaviour. `mutable` since they're scratch —
    // the formulation's logical state is unchanged.

    mutable Matrix<T> B_workspace_;        ///< strain_matrix output / stress_matrix scratch
    mutable Matrix<T> N_w_workspace_;      ///< displacement_shape_matrix output
    mutable Matrix<T> N_phi_workspace_;    ///< rotation_shape_matrix output
    mutable Matrix<T> N_sigma_workspace_;  ///< stress_matrix output
};


template <std::floating_point T, std::size_t d>
void
Element<T, d>::stress_matrix(const ElementValues<T, d>& ev) const
{
    strain_matrix(ev);
    const Index Q        = ev.results_[0].cols();
    const Index n_strain = B_workspace_.rows() / Q;
    const Index K        = B_workspace_.cols();

    N_sigma_workspace_.setZero(n_strain * Q, K);
    for (Index q = 0; q < Q; ++q) {
        const auto D = constitutive_matrix(ev, q);
        N_sigma_workspace_.middleRows(n_strain * q, n_strain).noalias() =
            D * B_workspace_.middleRows(n_strain * q, n_strain);
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::compute_local_stiffness(const ElementValues<T, d>& ev,
                                       Matrix<T>& stiffness) const
{
    strain_matrix(ev);
    const Matrix<T>& B   = B_workspace_;
    const Index Q        = ev.mapped_weights_.size();
    const Index n_strain = B.rows() / Q;
    const Index K        = B.cols();

    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        const T dV = ev.mapped_weights_(q) * ev.jac(q);
        const auto D = constitutive_matrix(ev, q);
        const auto B_q = B.middleRows(n_strain * q, n_strain);
        stiffness.noalias() += dV * (B_q.transpose() * D * B_q);
    }
}

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
