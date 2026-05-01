#ifndef PYCK_NITSCHE_CONDITION_HPP
#define PYCK_NITSCHE_CONDITION_HPP

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
 * @brief Nitsche method boundary condition.
 *
 * Enforces Dirichlet boundary conditions in a stable, consistent, symmetric
 * variational form without auxiliary unknowns.
 *
 * The condition is bound to a single (boundary, element, quadrature) site
 * with a length scale `thickness` used in the penalty term β/h. Multiple
 * physical fields enforced on that boundary are added via add(), each with
 * its own (displacement_field, traction_field, penalty, value) tuple.
 *
 * Per-span boundary/parent shape evaluation is performed once and reused
 * across all added terms.
 *
 * For each added term the variational form contributes:
 *   K += -∫ N^T · Q dΓ                         (consistency)
 *   K += -∫ Q^T · N dΓ                         (symmetry)
 *   K += +(β/h) ∫ N^T · N dΓ                   (penalty)
 *   F += +∫ Q^T · ū dΓ - (β/h) ∫ N^T · ū dΓ
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch / element parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class NitscheCondition : public Condition<T>
{
public:
    NitscheCondition(const PatchBoundary<T, d>& boundary,
                     const Element<T, d>& element,
                     const QuadratureRule<T, d - 1>& quadrature,
                     T thickness);

    /**
     * @brief Register a Nitsche term on this boundary.
     *
     * @param displacement_field BoundaryField for the solution variable (e.g. w, φ_n).
     * @param traction_field     BoundaryField for the traction work-conjugate to it.
     * @param penalty            Penalty parameter β; effective coefficient is β/h.
     * @param value              Prescribed value for the displacement field.
     */
    NitscheCondition& add(Ptr<const BoundaryField<T>> displacement_field,
                          Ptr<const BoundaryField<T>> traction_field,
                          T penalty,
                          T value = T(0));

    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

private:
    struct Term {
        Ptr<const BoundaryField<T>> displacement_field;
        Ptr<const BoundaryField<T>> traction_field;
        T penalty;
        T value;
    };

    const PatchBoundary<T, d>& boundary_;
    const Element<T, d>& element_;
    const QuadratureRule<T, d - 1>& quadrature_;

    T thickness_;
    std::vector<Term> terms_;
};

} // namespace pyck

#endif // PYCK_NITSCHE_CONDITION_HPP
