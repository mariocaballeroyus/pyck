#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <stdexcept>
#include <Eigen/Dense>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
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

    /// @brief Minimum required derivative order for shape-function evaluation.
    virtual std::size_t min_order() const { return 0; }

    // === Matrix Operators (Element Formulation-Agnostic) ============================

    /**
     * @brief Generalised-stress shape matrix S = D B. Writes into this
     *        element's own @ref N_sigma_workspace_; read the result via
     *        `element.N_sigma_workspace_` after the call. Internally
     *        reuses @ref B_workspace_ as scratch for B.
     */
    virtual void stress_matrix(const Patch<T, d>& patch,
                               const std::vector<Matrix<T>>& basis,
                               const IntrinsicGeometry<T, d>& ig) const;

    /**
     * @brief Local stiffness matrix \
     *        K = \int_{\Omega}{ B^T D B dV }.
     *
     * @param pv Per-element workspace bound via `ElementValues::reinit`. Carries
     *           the patch reference, basis values + derivatives, mapped
     *           quadrature weights, and flat element index.
     * @param stiffness The local stiffness matrix.
     */
    virtual void compute_local_stiffness(const ElementValues<T, d>& pv,
                                         Matrix<T>& stiffness) const;

    // === Matrix Operators (Element Formulation-Specific) ============================

    /**
     * @brief Strain-displacement operator B, row-stacked per qp. Writes
     *        into this element's own @ref B_workspace_; read the result
     *        via `element.B_workspace_` after the call. Implementations
     *        size the workspace via `setZero(rows, cols)` — Eigen's resize
     *        is a no-op when dimensions already match, so repeated calls
     *        on the same element reuse the existing storage.
     */
    virtual void strain_matrix(const Patch<T, d>& patch,
                               const std::vector<Matrix<T>>& basis,
                               const IntrinsicGeometry<T, d>& ig) const = 0;

    /**
     * @brief Constitutive operator D at quadrature point q.
     *
     * @param ig The intrinsic geometry.
     * @param q  Quadrature point index.
     * @return The constitutive operator (stack-resident ConstitutiveMatrix,
     *         sized at runtime to the formulation's strain dimension).
     */
    virtual ConstitutiveMatrix<T>
    constitutive_matrix(const IntrinsicGeometry<T, d>& ig, Index q) const = 0;

    // === Shape Matrices (Pure Virtual) ==============================================

    /**
     * @brief Transverse-displacement shape matrix N_w (Q × K). Writes into
     *        this element's own @ref N_w_workspace_; read the result via
     *        `element.N_w_workspace_` after the call.
     */
    virtual void displacement_shape_matrix(const Patch<T, d>& patch,
                                           const std::vector<Matrix<T>>& basis,
                                           const IntrinsicGeometry<T, d>& ig) const = 0;

    /**
     * @brief Rotation shape matrix N_φ. Writes into this element's own
     *        @ref N_phi_workspace_; read the result via
     *        `element.N_phi_workspace_` after the call.
     */
    virtual void rotation_shape_matrix(const Patch<T, d>& patch,
                                       const std::vector<Matrix<T>>& basis,
                                       const IntrinsicGeometry<T, d>& ig) const = 0;

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
Element<T, d>::stress_matrix(const Patch<T, d>& patch,
                             const std::vector<Matrix<T>>& basis,
                             const IntrinsicGeometry<T, d>& ig) const
{
    strain_matrix(patch, basis, ig);
    const Index Q        = basis[0].cols();
    const Index n_strain = B_workspace_.rows() / Q;
    const Index K        = B_workspace_.cols();

    N_sigma_workspace_.setZero(n_strain * Q, K);
    for (Index q = 0; q < Q; ++q) {
        const auto D = constitutive_matrix(ig, q);
        N_sigma_workspace_.middleRows(n_strain * q, n_strain).noalias() =
            D * B_workspace_.middleRows(n_strain * q, n_strain);
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::compute_local_stiffness(const ElementValues<T, d>& pv,
                                       Matrix<T>& stiffness) const
{
    const Patch<T, d>& patch          = pv.patch();
    const auto& basis                 = pv.results_;
    const auto& q_weights             = pv.mapped_weights_;
    const IntrinsicGeometry<T, d>& ig = pv.intrinsic_geometry_;

    strain_matrix(patch, basis, ig);
    const Matrix<T>& B   = B_workspace_;
    const Index Q        = q_weights.size();
    const Index n_strain = B.rows() / Q;
    const Index K        = B.cols();

    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        const T dV = q_weights(q) * ig.jac(q);
        const auto D = constitutive_matrix(ig, q);
        const auto B_q = B.middleRows(n_strain * q, n_strain);
        stiffness.noalias() += dV * (B_q.transpose() * D * B_q);
    }
}

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
