#ifndef PYCK_LOAD_BOUNDARY_CONDITION_HPP
#define PYCK_LOAD_BOUNDARY_CONDITION_HPP

#include <vector>
#include <Eigen/Dense>

#include "boundary_value.hpp"
#include "condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Natural (Neumann) boundary condition: integrates the variational
 *        external-work term
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class LoadBoundaryCondition : public Condition<T, d>
{
public:

    // === Constructors ===============================================================

    LoadBoundaryCondition(const PatchBoundary<T, d>& boundary,
                          const Element<T, d>& element,
                          const QuadratureRule<T, d - 1>& quadrature);

    // === Utility ====================================================================

    /**
     * @brief Register a constant traction conjugate to the field.
     *
     * @param field Projected field the traction is work-conjugate to.
     * @param value Prescribed value.
     */
    LoadBoundaryCondition& add(BoundaryValue<T> field, T value = T(0));

    /**
     * @brief Register pre-evaluated per-quadrature-point traction values.
     *
     * @param field          Projected field the traction is work-conjugate to.
     * @param values_at_qpts Pre-evaluated values at quadrature points.
     */
    LoadBoundaryCondition& add(BoundaryValue<T> field,
                               const Vector<T>& values_at_qpts);

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

    /// @brief Number of active boundary quadrature points.
    Index num_active_qpts() const;

private:

    /// @brief A single traction term registered on this boundary.
    struct Term {
        BoundaryValue<T> field;              ///< Work-conjugate value for the traction.
        bool varying = false;                ///< True if the traction varies per qp.
        T constant_value = T(0);             ///< Constant traction (when not varying).
        Vector<T> values_at_qpts;            ///< Per-qp traction; size num_active_qpts().
    };

    /// @brief Boundary edge the traction is applied on.
    const PatchBoundary<T, d>& boundary_;

    /// @brief Parent element formulation (supplies the field traces).
    const Element<T, d>& element_;

    /// @brief Boundary (d-1) quadrature rule.
    const QuadratureRule<T, d - 1>& quadrature_;

    /// @brief Registered traction terms.
    std::vector<Term> terms_;

};

} // namespace pyck

#endif // PYCK_LOAD_BOUNDARY_CONDITION_HPP
