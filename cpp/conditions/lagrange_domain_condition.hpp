#ifndef PYCK_LAGRANGE_DOMAIN_CONDITION_HPP
#define PYCK_LAGRANGE_DOMAIN_CONDITION_HPP

#include <Eigen/Dense>
#include <vector>

#include "condition.hpp"
#include "patch.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../assembly/dof_layout.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Domain Lagrange-multiplier condition.
 *
 * Weakly enforces a single scalar constraint of the form
 *
 *     ∫_Ω N_slot · u dΩ = value · |Ω|
 *
 * for each registered DOF slot, where N_slot picks the requested per-node
 * DOF index out of the element's primary basis.  Each registered slot
 * contributes one Lagrange multiplier λ to the augmented system; the
 * resulting saddle-point block is symmetric.
 *
 * Typical use case: suppress the rigid-body / zero-energy mode of an
 * un-pinned scalar DOF (e.g. ψ in PlateReissnerMindlinDispl2p) by
 * registering its slot with `value = 0`.
 *
 * @tparam T  Scalar floating-point type.
 * @tparam d  Patch / element parametric dimension.
 */
template <std::floating_point T, std::size_t d>
class LagrangeDomainCondition : public Condition<T>
{
public:
    LagrangeDomainCondition(const Patch<T, d>& patch,
                            const Element<T, d>& element,
                            const QuadratureRule<T, d>& quadrature);

    /**
     * @brief Register a DOF slot to constrain.
     *
     * @param dof_index  Per-node DOF index to extract (must satisfy
     *                   `dof_index < element.num_node_dofs()`).
     * @param value      Prescribed mean value of the field over Ω
     *                   (default 0, suppresses the constant mode).
     */
    LagrangeDomainCondition& add(std::size_t dof_index, T value = T(0));

    std::size_t num_dofs() const override;

    void allocate_dofs(DofLayout& layout,
                       const std::vector<DofLayout::BlockId>& primal_blocks) override;

    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               const std::vector<DofLayout::BlockId>& primal_blocks) const override;

private:
    struct Term {
        std::size_t dof_index;
        T value;
        DofLayout::BlockId block_id = 0; // assigned in allocate_dofs
    };

    const Patch<T, d>& patch_;
    const Element<T, d>& element_;
    const QuadratureRule<T, d>& quadrature_;
    std::vector<Term> terms_;
};

} // namespace pyck

#endif // PYCK_LAGRANGE_DOMAIN_CONDITION_HPP
