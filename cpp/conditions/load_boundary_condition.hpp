#ifndef PYCK_LOAD_BOUNDARY_CONDITION_HPP
#define PYCK_LOAD_BOUNDARY_CONDITION_HPP

#include <vector>
#include <Eigen/Dense>

#include "boundary_field.hpp"
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
     * @brief Register a traction to be enforced on this boundary.
     *
     * @param field Physical traction.
     * @param value Prescribed value.
     */
    LoadBoundaryCondition& add(Ptr<const BoundaryField<T>> field, T value);

    /**
     * @brief Register pre-evaluated per-quadrature-point traction values.
     * 
     * @param field Physical traction.
     * @param values_at_qpts Pre-evaluated values at quadrature points.
     */ 
    LoadBoundaryCondition& add(Ptr<const BoundaryField<T>> field,
                               const Vector<T>& values_at_qpts);

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

    /// @brief The patch to which the constrained boundary belongs to.
    const Patch<T, d>& patch() const override { return *boundary_.parent(); }

    /// @brief Number of active boundary quadrature points.
    Index num_active_qpts() const;

private:

    /// @brief A single traction term registered on this boundary.
    struct Term {
        Ptr<const BoundaryField<T>> field;   ///< Work-conjugate field for the traction.
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
