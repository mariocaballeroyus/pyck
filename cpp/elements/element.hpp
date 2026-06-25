#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include "patch.hpp"
#include "tensor_product.hpp"
#include "element_values.hpp"
#include "../types.hpp"

namespace pyck
{

// Forward declarations for the optional element-owned auxiliary-DOF hooks
// (allocate_dofs / apply). Defined in cpp/assembly/; only referenced by reference here.
class DofLayout;
template <std::floating_point T> class SystemAssembler;
template <std::floating_point T, std::size_t d> class PatchBlocks;

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
    TRACTION,      ///< Force resultants  [n¹¹ n²² n¹² (q¹ q²)] — conjugate to DISPLACEMENT.
    MOMENT,        ///< Moment resultants [m¹¹ m²² m¹²]         — conjugate to ROTATION.
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

    // === Optional auxiliary-DOF hooks (mixed formulations) ==========================
    //
    // An element may own auxiliary DOF blocks with their own connectivity (e.g. an
    // assumed-strain field on a coarser basis) that do not fit the per-node primal
    // layout. These mirror the `Condition` interface and are invoked by the assembler
    // once per patch-element: `allocate_dofs` in the allocation phase, `apply` in the
    // apply phase. Default no-ops, so standard elements are unaffected.

    /// @brief Allocate auxiliary DOF blocks this element owns. Default: no-op.
    virtual void allocate_dofs(DofLayout& layout) {}

    /// @brief Assemble this element's auxiliary cross-block contribution (e.g. a
    ///        mixed-formulation saddle) into the global system. Default: no-op.
    virtual void apply(SystemAssembler<T>& assembler, const DofLayout& layout,
                       const PatchBlocks<T, d>& blocks) const {}

    /// @brief Auxiliary-DOF contribution to a recovered field at parametric point @p pt,
    ///        gathered from the FULL solution by `Function` (the second of the two-block
    ///        gather). Fills @p N_aux (k × n_aux) and the matching GLOBAL DOF indices
    ///        @p aux_dofs for the chosen field; returns true if this element owns such a
    ///        contribution. Default: none, so standard elements keep the per-node path.
    virtual bool field_aux_contribution(const ElementValues<T, d>& /*ev*/,
                                        const ColMatrix<T, d>& /*pt*/, FieldType /*field*/,
                                        Matrix<T>& /*N_aux*/,
                                        std::vector<Index>& /*aux_dofs*/) const
    { return false; }

    // === Matrix Operators (Element Formulation-Agnostic) ============================

    /**
     * @brief Generalised-stress shape matrix S = D B. Writes into this
     *        element's own @ref N_sigma_; read the result via
     *        `element.N_sigma_` after the call. Internally
     *        reuses @ref B_voigt_ as scratch for B.
     */
    virtual void stress_shape_matrix(const ElementValues<T, d>& ev) const;

    /**
     * @brief Force-resultant shape matrix: the membrane-force and transverse-
     *        shear rows of @ref stress_shape_matrix, stacked per qp as
     *        `[n¹¹ n²² n¹² (q¹ q²)]` (shear absent for a shear-free shell). The
     *        static counterpart of @ref displacement_shape_matrix. Writes into
     *        @ref N_traction_.
     */
    virtual void traction_shape_matrix(const ElementValues<T, d>& ev) const;

    /**
     * @brief Moment-resultant shape matrix: the bending/twisting rows of
     *        @ref stress_shape_matrix, stacked per qp as `[m¹¹ m²² m¹²]`. The
     *        static counterpart of @ref rotation_shape_matrix. Writes into
     *        @ref N_moment_.
     */
    virtual void moment_shape_matrix(const ElementValues<T, d>& ev) const;

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
     * @brief Number of generalised-strain rows per quadrature point (8 for a
     *        membrane + bending + transverse-shear shell, 6 for a shear-free
     *        Kirchhoff-Love shell). Authoritative size for @ref strain_matrix.
     */
    virtual Index num_strains() const = 0;

    /**
     * @brief Strain-displacement operator B, row-stacked per qp, written into this
     *        element's own @ref B_voigt_.
     *
     * Non-virtual orchestrator: it sizes and zeros @ref B_voigt_ and stacks the
     * membrane / bending / transverse-shear sub-blocks below. A formulation
     * overrides only the blocks it changes — e.g. a mixed element zeroes the
     * membrane block and re-supplies it through an auxiliary field.
     */
    void strain_matrix(const ElementValues<T, d>& ev) const;

    /**
     * @brief Membrane / bending / transverse-shear sub-blocks of @ref strain_matrix.
     *
     * Each writes ONLY its interleaved rows (`num_strains()*q + offset`, offsets
     * 0 / 3 / 6) into the supplied buffer @p B, which the orchestrator has already
     * sized and zeroed — so a block never calls `setZero`. Default no-op so a
     * formulation lacking a block (Kirchhoff-Love transverse shear) contributes
     * nothing.
     */
    virtual void membrane_strain_matrix(const ElementValues<T, d>&, Matrix<T>&) const {}
    virtual void bending_strain_matrix (const ElementValues<T, d>&, Matrix<T>&) const {}
    virtual void shear_strain_matrix   (const ElementValues<T, d>&, Matrix<T>&) const {}

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

    // Cartesian mid-surface (bending) displacement v_b·dir: the first three primal DOF
    // slots, value-interpolated and projected onto `dir`, WITHOUT the formulation-
    // specific corrections the recovered `displacement` carries (e.g. the hierarchic
    // shear term w_s A_3 of the four-parameter shells). This is the bending surface that
    // a multipatch coupling ties, as opposed to the work-conjugate total displacement
    // that loads/BCs pair with. The base default reads slots 0..2 as the Cartesian
    // translation, which is correct for all the shells; for Kirchhoff–Love it coincides
    // with `displacement`.

    virtual void bending_displacement(const ElementValues<T, d>& parent,
                                      const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

    // Scalar trace of the hierarchic primary field ψ (the rotation-free shear
    // potential of the four-parameter shells): one row per qp, the nodal B-spline
    // placed in ψ's DOF slot. ψ is a scalar primary unknown, not a projected vector,
    // so there is no `dir`. The base default throws — a formulation must override to
    // declare it carries a ψ field.

    virtual void psi(const ElementValues<T, d>& parent, Matrix<T>& out) const;

    // Directional derivative of the hierarchic field ψ projected onto `dir`:
    // ∇ψ·dir = ψ_,α (A^α·dir), one scalar row per qp. With `dir` the boundary co-normal
    // this is the normal-slope trace ψ_,n a C1 coupling pairs across the seam (the C1
    // counterpart of the C0 `psi` value trace). Base throws — only the ψ-carrying shells
    // override it.

    virtual void psi_gradient(const ElementValues<T, d>& parent,
                              const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

    // Linearised geometric variation of the deformed surface director
    // a_3 = (a_1×a_2)/|a_1×a_2|, projected onto `dir`: δ(a_3·dir) as a (Q × K) row-
    // per-qp operator, linear in the dofs and evaluated about the reference state. This
    // is the surface-normal rotation built from the displacement gradient alone (a pure
    // geometry quantity, distinct from an independent director field). It is the trace a
    // multipatch C¹ / G1 director-continuity coupling pairs across the seam. Base throws;
    // a surface element overrides it (for the displacement-based shells it coincides with
    // the `rotation` trace, since the covariant tilt θ_α = −u_,α·A_3 are the components
    // of δa_3).

    virtual void director_variation(const ElementValues<T, d>& parent,
                                    const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

    // The work-conjugate *natural* trace: the boundary force traction t = σ·ν,
    // built from the generalised-stress shape S = D B and the in-surface boundary
    // co-normal ν, then projected onto `dir` — the consistency flux a Nitsche
    // condition pairs with the kinematic trace above. `conormal` is the outward
    // in-surface unit normal of the boundary edge (ν); `dir` selects the
    // work-conjugate component exactly as for `displacement`. The base default
    // covers the first-gradient (membrane + transverse-shear) force, exact for the
    // Reissner–Mindlin shells; a C¹ Kirchhoff–Love element would override to add the
    // effective-shear / corner contribution carried by the moment derivatives.

    virtual void traction(const ElementValues<T, d>& parent,
                          const ColMatrix<T, 3>& conormal,
                          const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

    // Augmented boundary traction: like the 4-argument overload, but also emits the
    // auxiliary-block flux columns `out_aux` (Q × n_aux) and their GLOBAL DOF indices
    // `aux_dofs` — e.g. a mixed element's ε̃-sourced membrane traction, which a Nitsche
    // condition scatters as the consistency coupling K(u_test, ε̃). The base default
    // forwards to the 4-argument flux and leaves the auxiliary part empty, so a standard
    // element's Nitsche block is unchanged.

    virtual void traction(const ElementValues<T, d>& parent, const ColMatrix<T, 3>& conormal,
                          const ColMatrix<T, 3>& dir, Matrix<T>& out, Matrix<T>& out_aux,
                          std::vector<Index>& aux_dofs) const
    {
        traction(parent, conormal, dir, out);
        out_aux.resize(parent.num_points(), 0);
        aux_dofs.clear();
    }

    // The moment counterpart of `traction`, work-conjugate to the `rotation` trace:
    // the boundary bending-moment vector m⃗ = mᵅᵝ ν_β A_α projected onto `dir`. Both
    // m⃗ and the rotation θ⃗ are in-surface (tangent) vectors, so this is a plain
    // tangential contraction — no transverse-shear / normal part. Same first-gradient
    // caveat as `traction`.

    virtual void moment(const ElementValues<T, d>& parent,
                        const ColMatrix<T, 3>& conormal,
                        const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

protected:

    /// @brief Contravariant surface components (d^1, d^2) of a 3D direction at qp
    ///        q: d^α = A^{αβ} (d · A_β). Pairs with the covariant tilt rows θ_α
    ///        for the rotation projections. (d == 2 only.)
    std::pair<T, T> contravariant_dir(const ElementValues<T, d>& parent, Index q,
                                      const Vector3<T>& dvec) const;

    /// @brief Kirchhoff–Love surface-normal variation δ(a_3·dir) on the first three
    ///        (Cartesian mid-surface) DOF slots of each node — θ_α = −u_,α·A_3, raised
    ///        and contracted with `dir`. Shared by the shells: the bending displacement
    ///        lives in slots 0..2 and drives the surface slope, while any trailing slots
    ///        (ψ, hierarchic rotations / difference vectors) carry shear, not surface
    ///        rotation, so they stay zero. Writes a (Q × num_node_dofs·N) operator.
    ///        (d == 2 only; a no-op size otherwise.)
    void surface_director_variation(const ElementValues<T, d>& parent,
                                    const ColMatrix<T, 3>& dir, Matrix<T>& out) const;

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
    mutable Matrix<T> N_sigma_;    ///< stress_shape_matrix output
    mutable Matrix<T> N_traction_; ///< traction_shape_matrix output (force resultants)
    mutable Matrix<T> N_moment_;   ///< moment_shape_matrix output (moment resultants)
    mutable Matrix<T> N_primal_;   ///< primal_shape_matrix output
};


template <std::floating_point T, std::size_t d>
void
Element<T, d>::strain_matrix(const ElementValues<T, d>& ev) const
{
    const Index Q    = ev.basis_derivs[0].cols();
    const Index N    = ev.basis_derivs[0].rows();
    const Index ns   = num_strains();
    const Index ndof = static_cast<Index>(num_node_dofs());

    B_voigt_.setZero(ns * Q, ndof * N);
    membrane_strain_matrix(ev, B_voigt_);
    bending_strain_matrix(ev, B_voigt_);
    shear_strain_matrix(ev, B_voigt_);
}

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
        const ConstitutiveMatrix<T> D = constitutive_matrix(ev, q);
        N_sigma_.middleRows(n_strain * q, n_strain).noalias() =
            D * B_voigt_.middleRows(n_strain * q, n_strain);
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::traction_shape_matrix(const ElementValues<T, d>& ev) const
{
    stress_shape_matrix(ev);
    if constexpr (d == 2) {
        const Index Q         = ev.basis_derivs[0].cols();
        const Index n_strain  = N_sigma_.rows() / Q;
        const Index K         = N_sigma_.cols();
        const bool  has_shear = (n_strain == 8);
        const Index n_force   = has_shear ? 5 : 3;

        N_traction_.setZero(n_force * Q, K);
        for (Index q = 0; q < Q; ++q) {
            N_traction_.middleRows(n_force * q, 3) =
                N_sigma_.middleRows(n_strain * q, 3);                  // membrane n^{αβ}
            if (has_shear)
                N_traction_.middleRows(n_force * q + 3, 2) =
                    N_sigma_.middleRows(n_strain * q + 6, 2);          // transverse shear q^α
        }
    } else {
        N_traction_.setZero(0, N_sigma_.cols());
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::moment_shape_matrix(const ElementValues<T, d>& ev) const
{
    stress_shape_matrix(ev);
    if constexpr (d == 2) {
        const Index Q        = ev.basis_derivs[0].cols();
        const Index n_strain = N_sigma_.rows() / Q;
        const Index K        = N_sigma_.cols();

        N_moment_.setZero(3 * Q, K);
        for (Index q = 0; q < Q; ++q)
            N_moment_.middleRows(3 * q, 3) =
                N_sigma_.middleRows(n_strain * q + 3, 3);             // bending/twisting m^{αβ}
    } else {
        N_moment_.setZero(0, N_sigma_.cols());
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
    Matrix<T> DB(n_strain, K);
    for (Index q = 0; q < Q; ++q) {
        const T dV = ev.mapped_weights_(q) * ev.jac(q);
        const ConstitutiveMatrix<T> D = constitutive_matrix(ev, q);
        const auto B_voigt_q = B.middleRows(n_strain * q, n_strain);
        DB.noalias() = D * B_voigt_q;
        stiffness.noalias() += dV * (B_voigt_q.transpose() * DB);
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
Element<T, d>::bending_displacement(const ElementValues<T, d>& parent,
                                    const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    // v_b·dir = Σ_{k<3} dir_k N^i: the Cartesian translation slots only, no shear
    // correction. (For KL this equals `displacement`; for the hierarchic shells it is
    // the bending surface without the recovered w_s A_3 term.)
    const Index Q    = parent.num_points();
    const Index N    = static_cast<Index>(parent.basis_derivs[0].rows());
    const Index ndof = static_cast<Index>(num_node_dofs());
    out.setZero(Q, ndof * N);
    for (Index q = 0; q < Q; ++q) {
        const auto Nf = parent.N(q);
        for (Index i = 0; i < N; ++i)
            for (Index k = 0; k < 3; ++k)
                out(q, ndof * i + k) = Nf(i) * dir(q, k);
    }
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

template <std::floating_point T, std::size_t d>
void
Element<T, d>::psi(const ElementValues<T, d>& /*parent*/, Matrix<T>& /*out*/) const
{
    throw std::logic_error(
        "Element::psi: this formulation has no hierarchic psi field. The psi trace "
        "is defined only for the four-parameter rotation-free shells.");
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::psi_gradient(const ElementValues<T, d>& /*parent*/,
                            const ColMatrix<T, 3>& /*dir*/, Matrix<T>& /*out*/) const
{
    throw std::logic_error(
        "Element::psi_gradient: this formulation has no hierarchic psi field. The psi "
        "normal-slope trace is defined only for the four-parameter rotation-free shells.");
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::director_variation(const ElementValues<T, d>& /*parent*/,
                                  const ColMatrix<T, 3>& /*dir*/, Matrix<T>& /*out*/) const
{
    throw std::logic_error(
        "Element::director_variation: this formulation does not supply a surface-director "
        "variation. It is implemented for the surface (shell) elements only.");
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::surface_director_variation(const ElementValues<T, d>& parent,
                                          const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    const Index Q    = parent.num_points();
    const Index N    = static_cast<Index>(parent.basis_derivs[0].rows());
    const Index ndof = static_cast<Index>(num_node_dofs());
    out.setZero(Q, ndof * N);
    if constexpr (d == 2) {
        for (Index q = 0; q < Q; ++q) {
            const Vector3<T> A3 = parent.normal(q);
            const Vector3<T> dq = dir.row(q).transpose();
            const auto [du1, du2] = contravariant_dir(parent, q, dq);  // d^α = A^α·dir
            const auto grad = parent.dN(q);
            for (Index i = 0; i < N; ++i) {
                // δa_3·dir = θ_α d^α with θ_α = −u_,α·A_3 on the Cartesian slots.
                const T coeff = -(du1 * grad(i, 0) + du2 * grad(i, 1));
                for (Index k = 0; k < 3; ++k)
                    out(q, ndof * i + k) = coeff * A3(k);
            }
        }
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::traction(const ElementValues<T, d>& parent,
                        const ColMatrix<T, 3>& conormal,
                        const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    stress_shape_matrix(parent);
    const Matrix<T>& S = N_sigma_;
    const Index Q        = parent.num_points();
    const Index n_strain = S.rows() / Q;
    out.resize(Q, S.cols());

    if constexpr (d == 2) {
        const bool has_shear = (n_strain == 8);
        for (Index q = 0; q < Q; ++q) {
            const auto A = parent.cov_basis(q);
            const Vector3<T> A1 = A(0), A2 = A(1), A3 = parent.normal(q);
            const Vector3<T> nu = conormal.row(q).transpose();
            const Vector3<T> dv = dir.row(q).transpose();
            const T nu1 = nu.dot(A1), nu2 = nu.dot(A2);
            const T d1  = dv.dot(A1), d2  = dv.dot(A2), d3 = dv.dot(A3);

            // Membrane: nᵅᵝ ν_β d_α, the symmetric {11,22,12} Voigt contraction.
            out.row(q) =  (nu1 * d1)            * S.row(n_strain * q + 0)
                        + (nu2 * d2)            * S.row(n_strain * q + 1)
                        + (nu2 * d1 + nu1 * d2) * S.row(n_strain * q + 2);

            // Transverse shear (Reissner–Mindlin): qᵅ ν_α resolved along A₃.
            if (has_shear)
                out.row(q) += (nu1 * d3) * S.row(n_strain * q + 6)
                            + (nu2 * d3) * S.row(n_strain * q + 7);
        }
    } else {
        out.setZero();
    }
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::moment(const ElementValues<T, d>& parent,
                      const ColMatrix<T, 3>& conormal,
                      const ColMatrix<T, 3>& dir, Matrix<T>& out) const
{
    stress_shape_matrix(parent);
    const Matrix<T>& S = N_sigma_;
    const Index Q        = parent.num_points();
    const Index n_strain = S.rows() / Q;
    out.resize(Q, S.cols());

    if constexpr (d == 2) {
        for (Index q = 0; q < Q; ++q) {
            const auto A = parent.cov_basis(q);
            const Vector3<T> A1 = A(0), A2 = A(1);
            const Vector3<T> nu = conormal.row(q).transpose();
            const Vector3<T> dv = dir.row(q).transpose();
            const T nu1 = nu.dot(A1), nu2 = nu.dot(A2);
            const T d1  = dv.dot(A1), d2  = dv.dot(A2);

            // Bending: mᵅᵝ ν_β d_α, the symmetric {11,22,12} Voigt contraction.
            out.row(q) =  (nu1 * d1)            * S.row(n_strain * q + 3)
                        + (nu2 * d2)            * S.row(n_strain * q + 4)
                        + (nu2 * d1 + nu1 * d2) * S.row(n_strain * q + 5);
        }
    } else {
        out.setZero();
    }
}

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
