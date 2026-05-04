#ifndef PYCK_LAGRANGE_COUPLING_CONDITION_HPP
#define PYCK_LAGRANGE_COUPLING_CONDITION_HPP

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
 * @brief Lagrange-multiplier coupling condition between two patches.
 *
 *        Bound to two boundaries (one from each coupled patch) and a single
 *        1D quadrature rule. Each registered field-pair (C_A, C_B, sign_b)
 *        gets its own Lagrange-multiplier block (one multiplier per side-A
 *        boundary control point) and contributes the symmetric saddle-point
 *        block
 *
 *            ∫_Γ μ · (C_A u_A − sB · C_B u_B − ḡ) dΓ = 0   ∀μ
 *
 *        where μ is approximated by side A's boundary basis. No penalty
 *        coefficient is required; the constraint is enforced exactly in
 *        the discrete multiplier space.
 *
 * @tparam T  Scalar floating-point type.
 * @tparam d  Parent patch parametric dimension (must be > 1).
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class LagrangeCouplingCondition : public Condition<T>
{
public:
    /**
     * @brief Construct a coupling condition between side A of patch index
     *        `patch_a_idx` and side B of patch index `patch_b_idx`.
     *
     * @param side_a       Boundary on side A.
     * @param side_b       Boundary on side B.
     * @param patch_a_idx  Index of patch A in the LinearElasticProblem.
     * @param patch_b_idx  Index of patch B.
     * @param element_a    Element formulation on patch A.
     * @param element_b    Element formulation on patch B.
     * @param quadrature   1D quadrature rule along the interface curve.
     * @param reverse      If true, side B's free-direction parametrisation
     *                     is the reverse of side A's.
     */
    LagrangeCouplingCondition(const PatchBoundary<T, d>& side_a,
                              const PatchBoundary<T, d>& side_b,
                              std::size_t patch_a_idx,
                              std::size_t patch_b_idx,
                              const Element<T, d>& element_a,
                              const Element<T, d>& element_b,
                              const QuadratureRule<T, d - 1>& quadrature,
                              bool reverse = false);

    /**
     * @brief Register a coupled physical field-pair.
     *
     * @param field_a  Field evaluated on side A.
     * @param field_b  Field evaluated on side B.
     * @param sign_b   Sign multiplier for side B (≈ ±1 depending on whether
     *                 the physical quantity flips across the interface, e.g.
     *                 +1 for transverse displacement, −1 for normal rotation
     *                 when outward normals are opposite).
     * @param value    Prescribed jump g̅ (default 0).
     */
    LagrangeCouplingCondition& add(Ptr<const BoundaryField<T>> field_a,
                                   Ptr<const BoundaryField<T>> field_b,
                                   T sign_b = T(1),
                                   T value  = T(0));

    std::size_t patch_a_idx() const { return patch_a_idx_; }
    std::size_t patch_b_idx() const { return patch_b_idx_; }
    bool reverse() const { return reverse_; }

    std::size_t num_dofs() const override;

    void allocate_dofs(DofLayout& layout,
                       const std::vector<DofLayout::BlockId>& primal_blocks) override;

    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               const std::vector<DofLayout::BlockId>& primal_blocks) const override;

private:
    struct Term {
        Ptr<const BoundaryField<T>> field_a;
        Ptr<const BoundaryField<T>> field_b;
        T sign_b;
        T value;
        DofLayout::BlockId block_id = 0; // assigned in allocate_dofs
    };

    const PatchBoundary<T, d>& side_a_;
    const PatchBoundary<T, d>& side_b_;
    std::size_t patch_a_idx_;
    std::size_t patch_b_idx_;
    const Element<T, d>& element_a_;
    const Element<T, d>& element_b_;
    const QuadratureRule<T, d - 1>& quadrature_;
    bool reverse_;

    Index multiplier_dof_count_ = 0;     // = side_a_.num_control_pts()
    std::vector<Term> terms_;
};

} // namespace pyck

#endif // PYCK_LAGRANGE_COUPLING_CONDITION_HPP
