#ifndef PYCK_MIXED_FORMULATION_HPP
#define PYCK_MIXED_FORMULATION_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "../conditions/condition.hpp"
#include "../elements/element.hpp"
#include "../geometry/patch.hpp"
#include "../quadrature/quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Base for Hellinger-Reissner mixed formulations that replace one strain
 *        block of a displacement element with an independent, globally-coupled
 *        field, kept sparse (no static condensation).
 *
 * @details The displacement element is told to `suppress_block` the governed
 *          strain block, so it assembles no stiffness for that block. The mixed
 *          field re-supplies the block through a two-field saddle (Guo, Zou &
 *          Ruess 2020; Bieber et al. 2018):
 *          \f[
 *            \begin{bmatrix} \mathbf K_{\text{rest}} & \mathbf G^\top\\
 *              \mathbf G & -\mathbf H \end{bmatrix},\quad
 *            \mathbf H=\!\int \tilde{\mathbf O}^\top \mathbf D_b \tilde{\mathbf O},\;
 *            \mathbf G=\!\int \tilde{\mathbf O}^\top \mathbf D_b \mathbf B_b,
 *          \f]
 *          where \f$\mathbf B_b\f$ / \f$\mathbf D_b\f$ are the governed block's rows of the
 *          displacement strain operator and constitutive matrix, and
 *          \f$\tilde{\mathbf O}\f$ is the subclass-supplied field operator (basis
 *          *values* for an assumed-strain method, *derivatives* for a
 *          mixed-displacement one).
 *
 *          Subclasses supply the governed block, the field's own DOF blocks, and
 *          the field operator; the base owns the assembly loop, the
 *          block-suppression wiring, and the saddle scatter.
 *
 * @note The displacement element's constitutive matrix must be block-diagonal
 *       across the governed block (no coupling to other strain blocks), so
 *       suppression removes exactly that block.
 *
 * @tparam T Scalar type.
 * @tparam d Patch / element parametric dimension.
 */
template <std::floating_point T, std::size_t d>
class MixedFormulation : public Condition<T, d>
{
public:

    /// @brief Allocate the field's DOF blocks and suppress the governed block on
    ///        the displacement element. Runs before the element loop.
    void allocate_dofs(DofLayout& layout) override;

    /// @brief Assemble the mixed saddle (G, Gᵀ, −H) into the global system.
    void apply(SystemAssembler<T>&       assembler,
               const DofLayout&          layout,
               const PatchBlocks<T, d>&  blocks) const final;

    const Patch<T, d>& patch() const final { return *patch_; }

protected:

    MixedFormulation(Ptr<Patch<T, d>>          patch,
                     Ptr<Element<T, d>>        element,
                     Ptr<QuadratureRule<T, d>> quadrature);

    /// @brief The strain block this formulation governs (e.g. membrane = {0, 3}).
    virtual StrainBlock governed_block() const = 0;

    /// @brief Allocate the auxiliary field's own DOF block(s). Called by
    ///        `allocate_dofs`.
    virtual void allocate_field_dofs(DofLayout& layout) = 0;

    /**
     * @brief Per element @p e: fill the stacked field operator @p O
     *        (`governed.count · Q` × `n_field`) and the global DOF indices
     *        @p dofs (`n_field`) of the auxiliary field on that element. Row block
     *        `[count·q, count·q + count)` of @p O is the operator at quadrature
     *        point @p q.
     */
    virtual void field_contribution(const ElementValues<T, d>& ev, Index e,
                                    const DofLayout& layout, Matrix<T>& O,
                                    std::vector<Index>& dofs) const = 0;

    Ptr<Patch<T, d>>          patch_;
    Ptr<Element<T, d>>        element_;
    Ptr<QuadratureRule<T, d>> quadrature_;
};

} // namespace pyck

#endif // PYCK_MIXED_FORMULATION_HPP
