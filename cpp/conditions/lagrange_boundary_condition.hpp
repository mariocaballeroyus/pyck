#ifndef PYCK_LAGRANGE_MULTIPLIER_BOUNDARY_CONDITION_HPP
#define PYCK_LAGRANGE_MULTIPLIER_BOUNDARY_CONDITION_HPP

#include <Eigen/Dense>
#include <vector>

#include "boundary_field.hpp"
#include "condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../assembly/dof_layout.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Boundary Lagrange-multiplier condition.
 *
 * The condition is bound to a single (boundary, element, quadrature) site.
 * Multiple physical fields enforced on that boundary are added via add(),
 * each receiving its own multiplier DOF block. Per-span boundary/parent
 * shape evaluation is performed once and reused across all added fields.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch / element parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class LagrangeBoundaryCondition : public Condition<T>
{
public:
    LagrangeBoundaryCondition(const PatchBoundary<T, d>& boundary,
                                        const Element<T, d>& element,
                                        const QuadratureRule<T, d - 1>& quadrature);

    /**
     * @brief Register a field to be enforced on this boundary.
     *
     * @param field  Physical field (e.g. transverse displacement, normal rotation).
     * @param value  Prescribed value.
     */
    LagrangeBoundaryCondition& add(Ptr<const BoundaryField<T>> field,
                                     T value = T(0));

    std::size_t num_dofs() const override;

    void allocate_dofs(DofLayout& layout, DofLayout::BlockId primal_block) override;

    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

private:
    struct Term {
        Ptr<const BoundaryField<T>> field;
        T value;
        DofLayout::BlockId block_id = 0; // assigned in allocate_dofs
    };

    const PatchBoundary<T, d>& boundary_;
    const Element<T, d>& element_;
    const QuadratureRule<T, d - 1>& quadrature_;

    std::vector<Term> terms_;
    Index multiplier_dof_count_ = 0;
};

} // namespace pyck

#endif // PYCK_LAGRANGE_MULTIPLIER_BOUNDARY_CONDITION_HPP
