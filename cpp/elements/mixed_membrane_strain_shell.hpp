#ifndef PYCK_MIXED_MEMBRANE_STRAIN_SHELL_HPP
#define PYCK_MIXED_MEMBRANE_STRAIN_SHELL_HPP

#include <array>
#include <vector>

#include "element.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "../geometry/dof_mapper.hpp"
#include "../assembly/dof_layout.hpp"
#include "../quadrature/quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Mixed assumed-membrane-strain shell (Guo, Zou & Ruess, IJNME 2020).
 *
 * @details A faithful, **global and uncondensed** Hellinger-Reissner treatment of
 *          membrane locking that wraps any displacement shell. The base element's
 *          membrane strain block is replaced at the source — `membrane_strain_matrix`
 *          is a no-op, so the assembled stiffness carries only bending + transverse
 *          shear (`K_bs`) and the recovered membrane stress is zero. The membrane is
 *          re-supplied through an independent strain field ε̃ = (ε̃₁₁, ε̃₂₂, 2ε̃₁₂) on
 *          three coarser anisotropic B-spline spaces (degrees (p-1,q), (p,q-1),
 *          (p-1,q-1); Echter Table 6.2 / Guo Eq. 29), kept as **global sparse**
 *          auxiliary DOFs (no static condensation): `allocate_dofs` allocates the
 *          field, `apply` assembles the sparse saddle [[-H, G], [Gᵀ, ·]] coupling it
 *          to the displacement DOFs.
 *
 *          The element *is* the element handed to the problem — bending, shear,
 *          constitutive and shape matrices forward to the base shell, so all
 *          displacement / rotation / moment recovery flows through it unchanged. The
 *          faithful membrane strain / force are recovered from the ε̃ field through the
 *          two-block gather (`field_aux_contribution`) and `recover_membrane_force`.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class MixedMembraneStrainShell : public Element<T, 2>
{
public:

    /**
     * @param patch       The patch carrying the displacement shell and the ε̃ field.
     * @param base        The base displacement shell (any shell element on @p patch).
     * @param quadrature  Quadrature rule for the mixed coupling (use the problem's).
     * @param degree_drop Degree reduction of the assumed-strain space (Guo: 1).
     */
    MixedMembraneStrainShell(Ptr<Patch<T, 2>>          patch,
                             Ptr<Element<T, 2>>        base,
                             Ptr<QuadratureRule<T, 2>> quadrature,
                             Index                     degree_drop = 1);

    // === DOF layout / formulation forwarding ========================================

    std::size_t num_node_dofs()  const override { return base_->num_node_dofs(); }
    Index       num_strains()    const override { return base_->num_strains(); }
    Index       basis_order()    const override { return base_->basis_order(); }
    unsigned    flags()          const override { return base_->flags(); }
    unsigned    essential_flags()const override { return base_->essential_flags(); }
    unsigned    natural_flags()  const override { return base_->natural_flags(); }

    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const override
    { return base_->constitutive_matrix(ev, q); }

    // === Strain blocks ==============================================================
    //
    // Membrane is suppressed at the source (no-op): the base orchestrator then yields a
    // B with zero membrane rows, so the inherited compute_local_stiffness gives K_bs and
    // the inherited stress_shape_matrix gives zero membrane stress. The saddle in `apply`
    // re-supplies the membrane. Bending / shear forward to the base, writing into B.

    void membrane_strain_matrix(const ElementValues<T, 2>&, Matrix<T>&) const override {}
    void bending_strain_matrix (const ElementValues<T, 2>& ev, Matrix<T>& B) const override
    { base_->bending_strain_matrix(ev, B); }
    void shear_strain_matrix   (const ElementValues<T, 2>& ev, Matrix<T>& B) const override
    { base_->shear_strain_matrix(ev, B); }

    // === Shape matrices (forward + copy into this element's own buffers) =============

    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override
    { base_->displacement_shape_matrix(ev); this->N_w_ = base_->N_w_; }
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override
    { base_->rotation_shape_matrix(ev); this->N_phi_ = base_->N_phi_; }

    using Element<T, 2>::traction;   // keep the 4-argument boundary flux visible

    /// @brief Augmented boundary traction: the per-node flux carries zero membrane
    ///        (suppressed) + the real transverse shear, while @p out_aux carries the
    ///        ε̃-sourced membrane traction (D_m·R̃ contracted with the conormal) over the
    ///        field's global DOF indices @p aux_dofs — the Nitsche consistency coupling.
    void traction(const ElementValues<T, 2>& parent, const ColMatrix<T, 3>& conormal,
                  const ColMatrix<T, 3>& dir, Matrix<T>& out, Matrix<T>& out_aux,
                  std::vector<Index>& aux_dofs) const override;

    // === Auxiliary (ε̃) DOF hooks ====================================================

    /// @brief Allocate the assumed-strain field's three DOF blocks. Runs before the
    ///        element loop.
    void allocate_dofs(DofLayout& layout) override;

    /// @brief Assemble the mixed saddle (G, Gᵀ, −H) into the global system.
    void apply(SystemAssembler<T>&      assembler,
               const DofLayout&         layout,
               const PatchBlocks<T, 2>& blocks) const override;

    /// @brief Auxiliary (ε̃) contribution to a recovered field at parametric point
    ///        @p pt, read from the full solution by @ref Function: the membrane rows of
    ///        the STRAIN (R̃) or TRACTION (D_m·R̃) shape, plus the matching global ε̃ DOF
    ///        indices. Returns false for fields the ε̃ field does not touch.
    bool field_aux_contribution(const ElementValues<T, 2>& ev, const ColMatrix<T, 2>& pt,
                                FieldType field, Matrix<T>& N_aux,
                                std::vector<Index>& aux_dofs) const override;

    // === Recovery ===================================================================

    /**
     * @brief Smooth membrane force resultant nᵅᵝ = D_m·ε̃ from the solved assumed
     *        strain field at parametric @p params (Q × 2), returning (Q × 3) Voigt
     *        resultants [n¹¹, n²², n¹²]. @p full_u is the untruncated solution
     *        (`ck.solve(problem, full=True)`).
     */
    Matrix<T> recover_membrane_force(const Vector<T>& full_u,
                                     const ColMatrix<T, 2>& params) const;

    /// @brief The wrapped base displacement element.
    const Element<T, 2>& base() const { return *base_; }

private:

    /// @brief Evaluate the assumed-strain field operator R̃ (3·Q × n_field) and its
    ///        global DOF indices on element @p e (its component spans on the coarser bases).
    void field_contribution(const ElementValues<T, 2>& ev, Index e,
                            const DofLayout& layout, Matrix<T>& O,
                            std::vector<Index>& dofs) const;

    Ptr<Patch<T, 2>>          patch_;
    Ptr<Element<T, 2>>        base_;
    Ptr<QuadratureRule<T, 2>> quadrature_;
    Index                     degree_drop_;

    /// @brief The three component bases (ε̃₁₁, ε̃₂₂, 2ε̃₁₂) and their DOF connectivity.
    std::array<Ptr<TensorProduct<T, 2>>, 3>   comp_basis_;
    std::array<DofMapper<2>, 3>               comp_mapper_;
    std::array<Index, 3>                      comp_ncp_;
    mutable std::array<DofLayout::BlockId, 3> comp_block_;
    mutable std::array<Index, 3>              comp_base_;   // global DOF offsets (set at allocate)
};

} // namespace pyck

#endif // PYCK_MIXED_MEMBRANE_STRAIN_SHELL_HPP
