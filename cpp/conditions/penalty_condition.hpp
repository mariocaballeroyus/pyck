#ifndef PYCK_PENALTY_CONDITION_HPP
#define PYCK_PENALTY_CONDITION_HPP

#include <Eigen/Dense>
#include <vector>

#include "boundary_field.hpp"
#include "condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Penalty method boundary condition.
 *
 * The condition is bound to a single (boundary, element, quadrature) site.
 * Multiple physical fields enforced on that boundary are added via add().
 * Per-span boundary/parent shape evaluation is performed once and reused
 * across all added fields.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch / element parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class PenaltyCondition : public Condition<T>
{
public:
    PenaltyCondition(const PatchBoundary<T, d>& boundary,
                     const Element<T, d>& element,
                     const QuadratureRule<T, d - 1>& quadrature);

    /**
     * @brief Register a field to be enforced on this boundary.
     *
     * @param field    Physical field (e.g. transverse displacement, normal rotation).
     * @param penalty  Penalty factor.
     * @param value    Prescribed value.
     */
    PenaltyCondition& add(Ptr<const BoundaryField<T>> field,
                          T penalty,
                          T value = T(0));

    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

private:
    struct Term {
        Ptr<const BoundaryField<T>> field;
        T penalty;
        T value;
    };

    const PatchBoundary<T, d>& boundary_;
    const Element<T, d>& element_;
    const QuadratureRule<T, d - 1>& quadrature_;

    std::vector<Term> terms_;
};

} // namespace pyck

#endif // PYCK_PENALTY_CONDITION_HPP
