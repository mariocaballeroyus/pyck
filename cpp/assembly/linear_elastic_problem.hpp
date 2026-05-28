#ifndef PYCK_LINEAR_ELASTIC_PROBLEM_HPP
#define PYCK_LINEAR_ELASTIC_PROBLEM_HPP

#include <vector>
#include <memory>
#include <concepts>
#include <stdexcept>

#include "patch.hpp"
#include "basis.hpp"
#include "element.hpp"
#include "condition.hpp"
#include "direct_constraint.hpp"
#include "constraint.hpp"
#include "dof_layout.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Global finite element linear elastic problem assembler.
 *
 *        Holds one or more NURBS patches, each with its own Element formulation
 *        and quadrature rule. Allocates one primal DOF block per patch in the
 *        global layout, iterates over the tensor-product elements of every
 *        patch, evaluates local element stiffness and force matrices, and
 *        scatters them into the global linear system. Boundary/load conditions
 *        are tagged with the patch index they belong to and applied per-block.
 *
 * @tparam T Scalar floating point type
 * @tparam d Parametric dimension of the Patch and Element
 */
template <std::floating_point T, std::size_t d>
class LinearElasticProblem
{
public:

    /**
     * @brief Construct an empty problem; patches are added later via add_patch().
     */
    LinearElasticProblem() = default;

    // Constructors ===================================================================

    /**
     * @brief Single-patch convenience constructor (backward compatible).
     *
     * @param patch The geometric patch over which to assemble.
     * @param element The finite element formulation.
     * @param quadrature The quadrature rule.
     */
    LinearElasticProblem(const Ptr<Patch<T, d>>& patch,
                         const Ptr<Element<T, d>>& element,
                         const Ptr<QuadratureRule<T, d>>& quadrature)
    {
        add_patch(patch, element, quadrature);
    }

    /**
     * @brief Multi-patch constructor: parallel vectors of patches/elements/quadratures.
     */
    LinearElasticProblem(const std::vector<Ptr<Patch<T, d>>>& patches,
                         const std::vector<Ptr<Element<T, d>>>& elements,
                         const std::vector<Ptr<QuadratureRule<T, d>>>& quadratures)
    {
        if (patches.size() != elements.size() || patches.size() != quadratures.size()) {
            throw std::invalid_argument(
                "LinearElasticProblem: patches, elements, and quadratures must "
                "have the same length.");
        }
        for (std::size_t p = 0; p < patches.size(); ++p) {
            add_patch(patches[p], elements[p], quadratures[p]);
        }
    }

    // === Customizers ================================================================

    /**
     * @brief Append a patch with its own element formulation and quadrature.
     *
     * @return The index of the newly added patch (0-based).
     */
    std::size_t add_patch(const Ptr<Patch<T, d>>& patch,
                          const Ptr<Element<T, d>>& element,
                          const Ptr<QuadratureRule<T, d>>& quadrature)
    {
        if (!patch || !element || !quadrature) {
            throw std::invalid_argument(
                "LinearElasticProblem::add_patch: null patch, element, or quadrature.");
        }
        patches_.push_back(patch);
        elements_.push_back(element);
        quadratures_.push_back(quadrature);
        conditions_per_patch_.emplace_back();
        return patches_.size() - 1;
    }

    /**
     * @brief Add a boundary/load condition.
     *
     *        The target patch is inferred from the condition's own patch by
     *        identity, so it must be one of the patches already registered
     *        with this problem.
     *
     * @param condition Shared pointer to the condition.
     */
    void add_condition(const Ptr<Condition<T, d>>& condition)
    {
        if (!condition) {
            throw std::invalid_argument(
                "LinearElasticProblem::add_condition: null condition.");
        }
        const Patch<T, d>* target = &condition->patch();
        for (std::size_t p = 0; p < patches_.size(); ++p) {
            if (patches_[p].get() == target) {
                conditions_per_patch_[p].push_back(condition);
                return;
            }
        }
        throw std::invalid_argument(
            "LinearElasticProblem::add_condition: the condition's patch is "
            "not registered with this problem.");
    }

    /**
     * @brief Add a constraint (e.g. LinearConstraint) to the system.
     *
     *        Constraints reference absolute global DOF indices and are applied
     *        after element + condition assembly.
     */
    void add_constraint(const Ptr<Constraint<T>>& constraint) {
        constraints_.push_back(constraint);
    }

    /**
     * @brief Add a strong direct boundary constraint.
     */
    void add_direct_constraint(const Ptr<DirectConstraint<T>>& constraint) {
        direct_constraints_.push_back(constraint);
    }

    // === Properties =================================================================

    /**
     * @brief Number of patches currently registered.
     */
    std::size_t num_patches() const { return patches_.size(); }

    /**
     * @brief Assemble the global stiffness matrix K and load vector F.
     *
     *        1. Allocates one primal DOF block per patch.
     *        2. Lets per-patch conditions allocate auxiliary DOFs.
     *        3. Loops over (patch, element) and scatters local stiffness.
     *        4. Applies per-patch conditions, then global constraints.
     */
    void assemble(Matrix<T>& K, Vector<T>& F) const;

private:

    std::vector<Ptr<Patch<T, d>>> patches_;

    std::vector<Ptr<Element<T, d>>> elements_;

    std::vector<Ptr<QuadratureRule<T, d>>> quadratures_;

    std::vector<std::vector<Ptr<Condition<T, d>>>> conditions_per_patch_;

    std::vector<Ptr<Constraint<T>>> constraints_;

    std::vector<Ptr<DirectConstraint<T>>> direct_constraints_;

    mutable DofLayout layout_;

};

} // namespace pyck

#endif // PYCK_LINEAR_ELASTIC_PROBLEM_HPP
