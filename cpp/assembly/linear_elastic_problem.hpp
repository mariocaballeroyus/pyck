#ifndef PYCK_LINEAR_ELASTIC_PROBLEM_HPP
#define PYCK_LINEAR_ELASTIC_PROBLEM_HPP

#include <vector>
#include <memory>
#include <concepts>

#include "patch.hpp"
#include "basis.hpp"
#include "element.hpp"
#include "condition.hpp"
#include "../constraints/dirichlet_bc.hpp"
#include "constraint.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Global finite element linear elastic problem assembler for a single Patch geometry.
 * 
 *        Iterates over the tensor-product elements (non-zero knot spans) of the patch, 
 *        evaluates local element stiffness and force matrices, and scatters them into 
 *        the global linear system. Boundary conditions are applied to the assembled system.
 * 
 * @tparam T Scalar floating point type
 * @tparam d Parametric dimension of the Patch and Element
 */
template <std::floating_point T, std::size_t d>
class LinearElasticProblem
{
public:

    /**
     * @brief Construct a new LinearElasticProblem object
     * 
     * @param patch The geometric patch over which to assemble.
     * @param element The finite element formulation (e.g. EulerBernoulliBeam1P).
     */
    LinearElasticProblem(const Ptr<Patch<T, d>>& patch,
              const Ptr<Element<T, d>>& element);

    /**
     * @brief Set the quadrature rule for numerical integration.
     * 
     * @param quadrature A 1D quadrature rule applied as a tensor product along each parametric direction.
     */
    void set_quadrature(const Ptr<QuadratureRule<T>>& quadrature);

    /**
     * @brief Add a boundary condition (e.g. LoadCondition, DisplacementCondition, RotationCondition) to the system.
     * 
     * @param condition Shared pointer to the condition.
     */
    void add_condition(const Ptr<Condition<T>>& condition);

    /**
     * @brief Add a constraint (e.g. MasterSlaveConstraint) to the system.
     * 
     * @param constraint Shared pointer to the constraint.
     */
    void add_constraint(const Ptr<Constraint<T>>& constraint);

    /**
     * @brief Add a strong Dirichlet boundary condition.
     * 
     * @param bc Shared pointer to the DirichletBC.
     */
    void add_dirichlet_bc(const Ptr<DirichletBC<T>>& bc);

    /**
     * @brief Assemble the global stiffness matrix K and load vector F.
     * 
     *        1. Initializes K to 0 and F to 0.
     *        2. Loops over all elements (non-zero knot spans).
     *        3. Adds element local contributions to the global system.
     *        4. Applies all registered Boundary/Load Conditions.
     * 
     * @param K The global stiffness matrix to assemble (output).
     * @param F The global load vector to assemble (output).
     */
    void assemble(Matrix<T>& K, Vector<T>& F) const;

private:

    Ptr<Patch<T, d>> patch_;

    Ptr<Element<T, d>> element_;

    Ptr<QuadratureRule<T>> quadrature_;

    std::vector<Ptr<Condition<T>>> conditions_;

    std::vector<Ptr<Constraint<T>>> constraints_;

    std::vector<Ptr<DirichletBC<T>>> dirichlet_bcs_;

};

} // namespace pyck

#endif // PYCK_LINEAR_ELASTIC_PROBLEM_HPP
