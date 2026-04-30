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
 * @brief Generic external load condition evaluating the integral: 
 *        f_ext = \int_{\Gamma} N^T t d\Gamma
 *        where N are the shape functions and t is the external force/traction.
 *
 * @tparam T Scalar type
 * @tparam d Parametric dimension
 */
template <std::floating_point T, std::size_t d>
class LoadCondition : public Condition<T>
{
public:
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

    /**
     * @brief Apply the integrated load to the global load vector.
     *
     * @param stiffness    Stiffness matrix
     * @param load         Load vector
     * @param layout       Equation-numbering authority.
     * @param primal_block Primal DOF block handle.
     */
    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

private:
    /// @brief Patch control-point indices contributing to the load.
    std::vector<Index> cp_indices_;

    /// @brief Element DOFs per control point.
    Index node_dofs_ = 0;

    /// @brief Load vector evaluated at the element level.
    Vector<T> element_load_;
};

} // namespace pyck

#endif // PYCK_LOAD_CONDITION_HPP