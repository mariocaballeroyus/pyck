#ifndef PYCK_PENALTY_COUPLING_CONDITION_HPP
#define PYCK_PENALTY_COUPLING_CONDITION_HPP

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
 * @brief Penalty-method coupling condition between two patches.
 *
 *        Bound to two boundaries (one from each coupled patch) and a single
 *        1D quadrature rule. Each registered field-pair (C_A, C_B, sign_b)
 *        contributes
 *
 *            K_pen = α ∫_Γ [  C_A^T C_A      − s_B C_A^T C_B ] dΓ
 *                          [ −s_B C_B^T C_A     C_B^T C_B    ]
 *
 *            F_pen = α ∫_Γ [   C_A^T  ḡ ] dΓ
 *                          [ −s_B C_B^T ḡ ]
 *
 *        scattered into the patch_a / patch_b primal blocks.
 *
 * @tparam T  Scalar floating-point type.
 * @tparam d  Parent patch parametric dimension (must be > 1).
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class PenaltyCouplingCondition : public Condition<T>
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
    PenaltyCouplingCondition(const PatchBoundary<T, d>& side_a,
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
     * @param penalty  Penalty factor α.
     * @param sign_b   Sign multiplier for side B (≈ ±1 depending on whether
     *                 the physical quantity flips across the interface, e.g.
     *                 +1 for transverse displacement, −1 for normal rotation
     *                 when outward normals are opposite).
     * @param value    Prescribed jump g̅ (default 0).
     */
    PenaltyCouplingCondition& add(Ptr<const BoundaryField<T>> field_a,
                                   Ptr<const BoundaryField<T>> field_b,
                                   T penalty,
                                   T sign_b = T(1),
                                   T value = T(0));

    std::size_t patch_a_idx() const { return patch_a_idx_; }
    std::size_t patch_b_idx() const { return patch_b_idx_; }
    bool reverse() const { return reverse_; }

    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               const std::vector<DofLayout::BlockId>& primal_blocks) const override;

private:
    struct Term {
        Ptr<const BoundaryField<T>> field_a;
        Ptr<const BoundaryField<T>> field_b;
        T penalty;
        T sign_b;
        T value;
    };

    const PatchBoundary<T, d>& side_a_;
    const PatchBoundary<T, d>& side_b_;
    std::size_t patch_a_idx_;
    std::size_t patch_b_idx_;
    const Element<T, d>& element_a_;
    const Element<T, d>& element_b_;
    const QuadratureRule<T, d - 1>& quadrature_;
    bool reverse_;

    std::vector<Term> terms_;
};

} // namespace pyck

#endif // PYCK_PENALTY_COUPLING_CONDITION_HPP
