#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <stdexcept>
#include <utility>
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
    PRIMAL,        ///< Primary DOFs                — N_p · u_local.
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
    ///        `compute_local_stiffness` / `compute_local_load` read. The bulk
    ///        assembler sizes the interior `ElementValues` workspace from it, so
    ///        it must also cover what `displacement_shape_matrix` reads (the
    ///        distributed-load path shares the interior workspace).
    virtual unsigned flags() const = 0;

    /// @brief Flags this element's *essential*-boundary shape matrices read
    ///        (`displacement_shape_matrix`, `rotation_shape_matrix`). Used by
    ///        kinematic boundary fields — those that constrain primary
    ///        unknowns (displacements, rotations).
    virtual unsigned essential_flags() const = 0;

    /// @brief Flags this element's `stress_shape_matrix` reads (= what its
    ///        `strain_matrix` + `constitutive_matrix` consume). Used by
    ///        *natural*-boundary fields — those applying work-conjugate
    ///        tractions (shear, bending moment, twisting moment).
    virtual unsigned natural_flags() const = 0;

    // === Matrix Operators (Element Formulation-Agnostic) ============================

    /**
     * @brief Generalised-stress shape matrix S = D B. Writes into this
     *        element's own @ref N_sigma_; read the result via
     *        `element.N_sigma_` after the call. Internally
     *        reuses @ref B_voigt_ as scratch for B.
     */
    virtual void stress_shape_matrix(const ElementValues<T, d>& ev) const;

    /**
     * @brief Primal-DOF shape matrix N_p.
     */
    virtual void primal_shape_matrix(const ElementValues<T, d>& ev) const;

    /**
     * @brief Local stiffness matrix K = ∫_Ω B^T D B dV.
     *
     * @param ev Per-element workspace bound via `ElementValues::reinit`.
     * @param stiffness The local stiffness matrix.
     */
    virtual void compute_local_stiffness(const ElementValues<T, d>& ev,
                                         Matrix<T>& stiffness) const;

    /**
     * @brief Local body-force load vector
     *
     * @param ev      Per-element workspace bound via `ElementValues::reinit`.
     * @param load_fn Body force per unit area: physical coords → force vectors.
     * @param f_local The local load vector.
     */
    void compute_local_domain_load(const ElementValues<T, d>& ev,
                                 const LoadFunction<T>& load_fn,
                                 Vector<T>& f_local) const;

    // === Matrix Operators (Element Formulation-Specific) ============================

    /**
     * @brief Strain-displacement operator B, row-stacked per qp. Writes
     *        into this element's own @ref B_voigt_.
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
     *        @ref N_w_.
     */
    virtual void displacement_shape_matrix(const ElementValues<T, d>& ev) const = 0;

    /**
     * @brief Rotation shape matrix N_φ. Writes into @ref N_phi_.
     */
    virtual void rotation_shape_matrix(const ElementValues<T, d>& ev) const = 0;

    // === Boundary Shape Traces ======================================================
    //
    // Trace of the recovered displacement / rotation projected onto a direction,
    // consumed by the boundary conditions: one scalar row per qp. The direction
    // `dir` (Q × 3 Cartesian, one per qp) is supplied by the caller — a global
    // axis ê_x/ê_y/ê_z for the Cartesian components, the boundary frame n / s for
    // the in-surface ones. A `Field` selection (U_X, U_N, ROT_S, …) is therefore
    // just a choice of `dir` in the `BoundaryValue` layer.
    //
    // Base defaults route through `displacement_shape_matrix` /
    // `rotation_shape_matrix`, so they are correct for every formulation. `dir`
    // is a raw `ColMatrix<T,3>` rather than a `BoundaryElementValues`, so these
    // stay boundary-agnostic and compile for every parametric dimension `d`.

    virtual void displacement(const ElementValues<T, d>& parent,
                              const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

    virtual void rotation(const ElementValues<T, d>& parent,
                          const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

protected:

    /// @brief Contravariant surface components (d^1, d^2) of a 3D direction at qp
    ///        q: d^α = A^{αβ} (d · A_β). Pairs with the covariant tilt rows θ_α
    ///        for the rotation projections. (d == 2 only.)
    std::pair<T, T> contravariant_dir(const ElementValues<T, d>& parent, Index q,
                                      const Vector3<T>& dvec) const;

public:

    // === Preallocated scratch =======================================================
    //
    // Per-formulation output matrices for the per-element shape/strain kernels. Public
    // because in-process callers (boundary fields, the assembler's hot loop)
    // write into them through a `const Element&` to get heap-free per-call
    // behaviour. `mutable` since they're scratch — the formulation's logical
    // state is unchanged.

    mutable Matrix<T> B_voigt_;        ///< strain_matrix output / stress_shape_matrix scratch
    mutable Matrix<T> N_w_;      ///< displacement_shape_matrix output
    mutable Matrix<T> N_phi_;    ///< rotation_shape_matrix output
    mutable Matrix<T> N_sigma_;  ///< stress_shape_matrix output
    mutable Matrix<T> N_primal_; ///< primal_shape_matrix output
};


template <std::floating_point T, std::size_t d>
void
Element<T, d>::stress_shape_matrix(const ElementValues<T, d>& ev) const
{
    strain_matrix(ev);
    const Index Q        = ev.basis_derivs[0].cols();
    const Index n_strain = B_voigt_.rows() / Q;
    const Index K        = B_voigt_.cols();

    N_sigma_.setZero(n_strain * Q, K);
    for (Index q = 0; q < Q; ++q) {
        const auto D = constitutive_matrix(ev, q);
        N_sigma_.middleRows(n_strain * q, n_strain).noalias() =
            D * B_voigt_.middleRows(n_strain * q, n_strain);
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::primal_shape_matrix(const ElementValues<T, d>& ev) const
{
    const Index Q    = ev.basis_derivs[0].cols();
    const Index N    = ev.basis_derivs[0].rows();
    const Index ndof = static_cast<Index>(num_node_dofs());
    Matrix<T>& Np    = N_primal_;
    Np.setZero(ndof * Q, ndof * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.basis_derivs[0].col(q);
        for (Index i = 0; i < N; ++i)
            for (Index v = 0; v < ndof; ++v)
                Np(ndof * q + v, i * ndof + v) = slab0(i);
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::compute_local_stiffness(const ElementValues<T, d>& ev,
                                       Matrix<T>& stiffness) const
{
    strain_matrix(ev);
    const Matrix<T>& B   = B_voigt_;
    const Index Q        = ev.mapped_weights_.size();
    const Index n_strain = B.rows() / Q;
    const Index K        = B.cols();

    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        const T dV = ev.mapped_weights_(q) * ev.jac(q);
        const auto D = constitutive_matrix(ev, q);
        const auto B_voigt_q = B.middleRows(n_strain * q, n_strain);
        stiffness.noalias() += dV * (B_voigt_q.transpose() * D * B_voigt_q);
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::compute_local_domain_load(const ElementValues<T, d>& ev,
                                       const LoadFunction<T>& load_fn,
                                       Vector<T>& f_local) const
{
    // f_local = \int{ U^T b d\Omega }, with U the 3D displacement shape (3Q × K).
    displacement_shape_matrix(ev);
    const Matrix<T>& U = N_w_;

    // Body force at the element's physical quadrature-point coordinates (Q × 3).
    const Matrix<T> b = load_fn(ev.position_derivs[0]);

    const Index Q = ev.mapped_weights_.size();
    f_local.setZero(U.cols());
    for (Index q = 0; q < Q; ++q) {
        const T dV = ev.mapped_weights_(q) * ev.jac(q);
        f_local.noalias() += dV *
            (U.middleRows(3 * q, 3).transpose() * b.row(q).transpose());
    }
}

// === Boundary Shape Traces ==========================================================

template <std::floating_point T, std::size_t d>
std::pair<T, T>
Element<T, d>::contravariant_dir(const ElementValues<T, d>& parent, Index q,
                                 const Vector3<T>& dvec) const
{
    const auto A = parent.cov_basis(q);
    const Vector3<T> A1 = A(0), A2 = A(1);
    const T dc1 = dvec.dot(A1), dc2 = dvec.dot(A2);
    const T gi11 = parent.metric_inv(q, 0);
    const T gi22 = parent.metric_inv(q, 1);
    const T gi12 = parent.metric_inv(q, 2);
    return { gi11 * dc1 + gi12 * dc2, gi12 * dc1 + gi22 * dc2 };
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::displacement(const ElementValues<T, d>& parent,
                            const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    // u·dir from the recovered Cartesian displacement shape (3 rows/qp = N_w_).
    displacement_shape_matrix(parent);
    const Matrix<T>& Nw = N_w_;
    const Index Q = parent.num_points();
    out.resize(Q, Nw.cols());
    for (Index q = 0; q < Q; ++q)
        out.row(q).noalias() = dir.row(q) * Nw.middleRows(3 * q, 3);
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::rotation(const ElementValues<T, d>& parent,
                        const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    // θ·dir = d^α θ_α: contravariant raise of `dir` contracted with the covariant
    // tilt rows (2/qp = N_phi_).
    rotation_shape_matrix(parent);
    const Matrix<T>& Np = N_phi_;
    const Index Q = parent.num_points();
    out.resize(Q, Np.cols());
    if constexpr (d == 2) {
        for (Index q = 0; q < Q; ++q) {
            const Vector3<T> dq = dir.row(q).transpose();
            const auto [du1, du2] = contravariant_dir(parent, q, dq);
            out.row(q) = du1 * Np.row(2 * q) + du2 * Np.row(2 * q + 1);
        }
    } else {
        out.setZero();
    }
}

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
