#ifndef PYCK_BOUNDARY_LAGRANGE_CONDITION_HPP
#define PYCK_BOUNDARY_LAGRANGE_CONDITION_HPP

#include <Eigen/Dense>
#include <vector>

#include "boundary_field.hpp"
#include "condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "dof_layout.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Boundary Lagrange-multiplier condition.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class LagrangeBoundaryCondition : public Condition<T, d>
{
public:

    // === Constructors ===============================================================

    LagrangeBoundaryCondition(const PatchBoundary<T, d>& boundary,
                              const Element<T, d>& element,
                              const QuadratureRule<T, d - 1>& quadrature);

    // === Utility ====================================================================

    /**
     * @brief Register a field to be enforced on this boundary.
     *
     * @param field  Physical field (e.g. transverse displacement, normal rotation).
     * @param value  Prescribed value.
     */
    LagrangeBoundaryCondition& add(Ptr<const BoundaryField<T>> field,
                                   T value = T(0));

    /**
     * @brief Allocate auxiliary DOF blocks. Default: no-op.
     *
     * @param layout Equation-numbering authority (mutable).
     */
    void allocate_dofs(DofLayout& layout) override;

    /**
     * @brief Assemble the condition's contribution into the global system.
     *
     * @param assembler     Sparse-assembly sink for stiffness and load (mutable).
     * @param layout        DOF layout (read-only).
     * @param primal_block  This condition's patch primal block.
     */
    void apply(SystemAssembler<T>& assembler,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

    // === Properties =================================================================

    /// @brief The patch to which the constrained boundary belongs to.
    const Patch<T, d>& patch() const override { return *boundary_.parent(); }

private:

    /// @brief A single field constraint registered on this boundary.
    struct Term {
        Ptr<const BoundaryField<T>> field;   ///< Field whose trace is constrained.
        T value;                             ///< Prescribed value on the boundary.
        DofLayout::BlockId block_id = 0;     ///< Multiplier block, set in allocate_dofs.
    };

    /// @brief Registered field constraints; one multiplier block per term.
    std::vector<Term> terms_;

    /// @brief Multiplier DOFs per term (= number of boundary control points).
    Index multiplier_dof_count_ = 0;

    /// @brief Boundary edge the constraint is enforced on.
    const PatchBoundary<T, d>& boundary_;

    /// @brief Parent element formulation (supplies the field traces).
    const Element<T, d>& element_;

    /// @brief Boundary (d-1) quadrature rule.
    const QuadratureRule<T, d - 1>& quadrature_;

};

} // namespace pyck

#endif // PYCK_BOUNDARY_LAGRANGE_CONDITION_HPP
