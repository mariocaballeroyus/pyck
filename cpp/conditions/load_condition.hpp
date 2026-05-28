#ifndef PYCK_LOAD_CONDITION_HPP
#define PYCK_LOAD_CONDITION_HPP


#include <vector>
#include <functional>
#include <Eigen/Dense>

#include "condition.hpp"
#include "patch.hpp"
#include "quadrature.hpp"
#include "element.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Generic domain load condition.
 *
 * @tparam T Scalar type
 * @tparam d Parametric dimension
 */
template <std::floating_point T, std::size_t d>
class LoadCondition : public Condition<T, d>
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Construct a generic load condition by numerically integrating the
     *        load values.
     *
     * @param patch The geometric patch over which to integrate.
     * @param element The element formulation.
     * @param quadrature The 1D quadrature rule.
     * @param load_values Pre-evaluated load values at the quadrature points.
     */
    LoadCondition(const Patch<T, d>& patch,
                  const Element<T, d>& element,
                  const QuadratureRule<T, d>& quadrature,
                  const Vector<T>& load_values);

    // === Utility ====================================================================

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

    /// @brief The patch where the domain load condition is applied. 
    const Patch<T, d>& patch() const override { return patch_; }

private:

    /// @brief Patch this load is integrated over (for DOF-block routing).
    const Patch<T, d>& patch_;

    /// @brief Patch control-point indices contributing to the load.
    std::vector<Index> cp_indices_;

    /// @brief Element DOFs per control point.
    Index node_dofs_ = 0;

    /// @brief Load vector evaluated at the element level.
    Vector<T> element_load_;

};

} // namespace pyck

#endif // PYCK_LOAD_CONDITION_HPP