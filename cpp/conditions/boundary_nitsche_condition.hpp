#ifndef PYCK_BOUNDARY_NITSCHE_CONDITION_HPP
#define PYCK_BOUNDARY_NITSCHE_CONDITION_HPP

#include <Eigen/Dense>
#include <vector>

#include "boundary_value.hpp"
#include "condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Symmetric Nitsche boundary condition.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class NitscheBoundaryCondition : public Condition<T, d>
{
public:

    // === Constructors ===============================================================

    NitscheBoundaryCondition(const PatchBoundary<T, d>& boundary,
                             const Element<T, d>& element,
                             const QuadratureRule<T, d - 1>& quadrature);

    // === Utility ====================================================================

    /**
     * @brief Register a boundary value to be enforced on this boundary.
     *
     * @param field    Boundary value (e.g. displacement / rotation projection).
     * @param penalty  Stabilisation factor α.
     * @param value    Prescribed value.
     */
    NitscheBoundaryCondition& add(BoundaryValue<T> field,
                                  T penalty, T value = T(0));

    /**
     * @brief Assemble the condition's contribution into the global system.
     *
     * @param assembler  Sparse-assembly sink for stiffness and load (mutable).
     * @param layout     DOF layout (read-only).
     * @param blocks     Patch to primal-block resolver.
     */
    void apply(SystemAssembler<T>& assembler,
               const DofLayout& layout,
               const PatchBlocks<T, d>& blocks) const override;

    // === Properties =================================================================

    /// @brief The patch to which the constrained boundary belongs.
    const Patch<T, d>& patch() const override { return *boundary_.parent(); }

private:
    /// @brief A single Nitsche-enforced field constraint registered on this boundary.
    struct Term {
        BoundaryValue<T> field;              ///< Boundary value whose trace is enforced.
        T penalty;                           ///< Stabilisation weight α.
        T value;                             ///< Prescribed value on the boundary.
    };

    /// @brief Boundary edge the constraint is enforced on.
    const PatchBoundary<T, d>& boundary_;

    /// @brief Parent element formulation (supplies the field and flux traces).
    const Element<T, d>& element_;

    /// @brief Boundary (d-1) quadrature rule.
    const QuadratureRule<T, d - 1>& quadrature_;

    /// @brief Registered field constraints.
    std::vector<Term> terms_;

};

} // namespace pyck

#endif // PYCK_BOUNDARY_NITSCHE_CONDITION_HPP