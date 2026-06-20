#ifndef PYCK_MEMBRANE_ASSUMED_STRAIN_CONDITION_HPP
#define PYCK_MEMBRANE_ASSUMED_STRAIN_CONDITION_HPP

#include <array>
#include <vector>

#include "condition.hpp"
#include "../basis/tensor_product.hpp"
#include "../geometry/dof_mapper.hpp"
#include "../geometry/patch.hpp"
#include "../elements/element.hpp"
#include "../quadrature/quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Guo, Zou & Ruess (IJNME 2020) assumed-membrane-strain (Hellinger-Reissner)
 *        treatment of membrane locking, kept **global and sparse** (no condensation).
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class MembraneAssumedStrainCondition : public Condition<T, 2>
{
public:

    /**
     * @param patch       The patch carrying the displacement shell.
     * @param element     The displacement shell element (same instance used in the problem,
     *                    so its membrane suppression takes effect).
     * @param quadrature  Quadrature rule for the mixed coupling (use the problem's).
     * @param degree_drop Degree reduction of the assumed strain (Guo: 1).
     */
    MembraneAssumedStrainCondition(Ptr<Patch<T, 2>>          patch,
                                   Ptr<Element<T, 2>>        element,
                                   Ptr<QuadratureRule<T, 2>> quadrature,
                                   Index                     degree_drop = 1);

    /// @brief Allocate the assumed-strain field's DOF blocks and suppress the membrane
    ///        block on the displacement element. Runs before the element loop.
    void allocate_dofs(DofLayout& layout) override;

    /// @brief Assemble the mixed saddle (G, Gᵀ, −H) into the global system.
    void apply(SystemAssembler<T>&       assembler,
               const DofLayout&          layout,
               const PatchBlocks<T, 2>&  blocks) const override;

    const Patch<T, 2>& patch() const override { return *patch_; }

    /**
     * @brief Recover the membrane force resultant nᵅᵝ = D_m·ε̃ from the solved assumed
     *        strain field at parametric @p params (Q × 2), returning (Q × 3) Voigt
     *        resultants [n¹¹, n²², n¹²]. @p full_u is the **untruncated** solution
     *        (including the ε̃ auxiliary blocks; `ck.solve(problem, full=True)`) — the
     *        smooth, locking-free membrane force, recovered from the field rather than the
     *        (locked) displacement gradient.
     */
    Matrix<T> recover_membrane_force(const Vector<T>& full_u,
                                     const ColMatrix<T, 2>& params) const;

private:

    /// @brief Per element @p e: fill the stacked field-value operator @p O (3·Q × n_field)
    ///        and the global DOF indices @p dofs of the assumed-strain field on that element.
    void field_contribution(const ElementValues<T, 2>& ev, Index e,
                            const DofLayout& layout, Matrix<T>& O,
                            std::vector<Index>& dofs) const;

    Ptr<Patch<T, 2>>          patch_;
    Ptr<Element<T, 2>>        element_;
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

#endif // PYCK_MEMBRANE_ASSUMED_STRAIN_CONDITION_HPP
