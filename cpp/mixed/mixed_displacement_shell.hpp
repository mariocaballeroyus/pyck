#ifndef PYCK_MIXED_DISPLACEMENT_SHELL_HPP
#define PYCK_MIXED_DISPLACEMENT_SHELL_HPP

#include <concepts>
#include <cstddef>

#include "mixed_displacement_kernels.hpp"
#include "../elements/element.hpp"
#include "../elements/element_values.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Mixed-Displacement (MD) membrane-locking treatment applied to any base shell.
 *
 * @details A composition decorator: it wraps a base displacement element and enriches it
 *          with the 3-component equal-order membrane strain-displacement field of the MD
 *          method (Bieber et al. 2018), supplied by the `md::` kernels. The result is one
 *          class that works for *every* base shell — there is no per-shell MD subclass.
 *
 *          The combined node layout is `[ base DOFs (ndof_base) ; ũ (3) ]`, so the ũ slots
 *          are the **last three** per node. Like the displacement element, the ũ field has
 *          integration-constant zero-energy modes that must be stabilised by Dirichlet pins
 *          on its boundary DOFs (see the Python helper `membrane_md_boundary_dofs`).
 *
 * @note The base element's `constitutive_matrix` must be block-diagonal (membrane / bending /
 *       shear), as the kernels read those blocks independently.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class MixedDisplacementShell : public Element<T, 2>
{
public:

    /// @brief Wrap a base shell element. Throws if @p base is null.
    explicit MixedDisplacementShell(Ptr<Element<T, 2>> base);

    // === DOF layout =================================================================

    /// @brief Base DOFs plus the three membrane strain-displacements.
    std::size_t num_node_dofs() const override
    { return base_->num_node_dofs() + md::kMembrane; }

    Index    basis_order()     const override { return base_->basis_order(); }
    unsigned flags()           const override { return base_->flags(); }
    unsigned essential_flags() const override { return base_->essential_flags(); }
    unsigned natural_flags()   const override { return base_->natural_flags(); }

    // === Matrix Operators ===========================================================

    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const override
    { return base_->constitutive_matrix(ev, q); }

    /// @brief Strain B in the combined layout. Used only by `FieldType::STRAIN` recovery:
    ///        the stiffness/stress kernels read the *base* strain via `*base_` directly. The
    ///        ũ columns are zero, so this reports the base (displacement) strain only.
    void strain_matrix(const ElementValues<T, 2>& ev) const override
    {
        base_->strain_matrix(ev);
        this->B_voigt_ = base_->B_voigt_;
        md::widen_dof_columns<T>(this->B_voigt_, ndof_base(), num_basis(ev));
    }

    void compute_local_stiffness(const ElementValues<T, 2>& ev, Matrix<T>& stiffness) const override
    { md::assemble_local_stiffness<T>(*base_, ndof_base(), ev, stiffness); }

    // === Shape Matrices =============================================================

    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        base_->displacement_shape_matrix(ev);
        this->N_w_ = base_->N_w_;
        md::widen_dof_columns<T>(this->N_w_, ndof_base(), num_basis(ev));
    }

    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override
    {
        base_->rotation_shape_matrix(ev);
        this->N_phi_ = base_->N_phi_;
        md::widen_dof_columns<T>(this->N_phi_, ndof_base(), num_basis(ev));
    }

    /// @brief Membrane stress from the ũ field, bending/shear from the displacement; drives
    ///        the (default) traction / moment boundary traces and recovered resultants.
    void stress_shape_matrix(const ElementValues<T, 2>& ev) const override
    { md::assemble_stress_shape<T>(*base_, ndof_base(), ev, this->N_sigma_); }

    // === Getters ====================================================================

    /// @brief The wrapped base element.
    const Element<T, 2>& base() const { return *base_; }

private:

    Index ndof_base() const { return static_cast<Index>(base_->num_node_dofs()); }

    static Index num_basis(const ElementValues<T, 2>& ev)
    { return static_cast<Index>(ev.basis_derivs[0].rows()); }

    Ptr<Element<T, 2>> base_;
};

} // namespace pyck

#endif // PYCK_MIXED_DISPLACEMENT_SHELL_HPP
