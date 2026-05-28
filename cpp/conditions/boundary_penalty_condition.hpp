#ifndef PYCK_BOUNDARY_PENALTY_CONDITION_HPP
#define PYCK_BOUNDARY_PENALTY_CONDITION_HPP

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
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class PenaltyBoundaryCondition : public Condition<T, d>
{
public:

    // === Constructors ===============================================================

    PenaltyBoundaryCondition(const PatchBoundary<T, d>& boundary,
                             const Element<T, d>& element,
                             const QuadratureRule<T, d - 1>& quadrature);

    // === Utility ====================================================================

    /**
     * @brief Register a field to be enforced on this boundary.
     *
     * @param field    Physical field (e.g. transverse displacement, normal rotation).
     * @param penalty  Penalty factor.
     * @param value    Prescribed value.
     */
    PenaltyBoundaryCondition& add(Ptr<const BoundaryField<T>> field,
                                  T penalty, T value = T(0));

                 
    /**
     * @brief Assemble the condition's contribution into the global system.
     *
     * @param stiffness     Global stiffness matrix (mutable).
     * @param load          Global load vector (mutable).
     * @param layout        DOF layout (read-only).
     * @param primal_block  This condition's patch primal block.
     */
    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

    // === Properties =================================================================

    /// @brief The patch to which the constrained boundary belongs.
    const Patch<T, d>& patch() const override { return *boundary_.parent(); }

private:
    /// @brief A single penalised field constraint registered on this boundary.
    struct Term {
        Ptr<const BoundaryField<T>> field;   ///< Field whose trace is penalised.
        T penalty;                           ///< Penalty weight α.
        T value;                             ///< Prescribed value on the boundary.
    };

    /// @brief Boundary edge the penalty is enforced on.
    const PatchBoundary<T, d>& boundary_;

    /// @brief Parent element formulation (supplies the field traces).
    const Element<T, d>& element_;

    /// @brief Boundary (d-1) quadrature rule.
    const QuadratureRule<T, d - 1>& quadrature_;

    /// @brief Registered penalised field constraints.
    std::vector<Term> terms_;

};

} // namespace pyck

#endif // PYCK_BOUNDARY_PENALTY_CONDITION_HPP
