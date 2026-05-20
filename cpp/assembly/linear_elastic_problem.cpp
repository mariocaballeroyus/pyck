#include "linear_elastic_problem.hpp"
#include "bspline.hpp"
#include "direct_constraint.hpp"
#include "patch.hpp"
#include "values.hpp"

#include <stdexcept>
#include <string>

namespace pyck
{

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::assemble(Matrix<T>& K, Vector<T>& F) const
{
    if (patches_.empty()) {
        throw std::runtime_error("LinearElasticProblem::assemble: no patches registered.");
    }

    // Reset DOF layount
    layout_.clear();
    std::vector<DofLayout::BlockId> primal_blocks;
    primal_blocks.reserve(patches_.size());

    // Allocate primal DOFs per patch
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        if (!quadratures_[p]) {
            throw std::runtime_error("LinearElasticProblem::assemble: "
                                     "quadrature rule not set for patch " + std::to_string(p) + ".");
        }

        const std::size_t ndof = elements_[p]->num_node_dofs();
        const std::size_t num_cps = patches_[p]->num_control_pts();
        primal_blocks.push_back(layout_.allocate(DofType::Primal, num_cps * ndof, ndof));
    }

    // Allocate auxiliary DOFs per patch (e.g. Lagrange multipliers).
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        for (const auto& cond : conditions_per_patch_[p]) {
            cond->allocate_dofs(layout_, primal_blocks);
        }
    }

    // Reset stiffness and load
    const std::size_t total_dofs = layout_.num_dofs();
    K.setZero(total_dofs, total_dofs);
    F.setZero(total_dofs);

    // Single allocation of element stiffness
    Matrix<T> Ke;

    // === Patch Loop =================================================================
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        const auto& patch = *patches_[p];
        const auto& element = *elements_[p];
        const auto& quadrature = *quadratures_[p];
        const auto& mapper = patch.dof_mapper();
        const DofLayout::BlockId primal_block = primal_blocks[p];

        // Total number of candidate elements in the patch (includes zero-volume
        // spans; PatchValues::reinit filters them).
        std::size_t total_elements = 1;
        const auto& tp_intervals = patch.tensor_product().num_intervals();
        for (std::size_t i = 0; i < d; ++i) {
            total_elements *= tp_intervals[i];
        }

        PatchValues<T, d> patch_values(patch, element, quadrature);

        // === Element Loop ===========================================================
        for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx) {
            if (!patch_values.reinit(elem_idx)) continue;

            element.compute_local_stiffness(patch, elem_idx,
                                            patch_values.basis_out,
                                            patch_values.mapped_weights, Ke);

            auto elem_nodes = mapper.get_element_dofs(elem_idx);
            auto elem_dofs  = layout_.scatter_primal(primal_block, elem_nodes);
            const std::size_t Ne = elem_dofs.size();
            for (std::size_t i = 0; i < Ne; ++i) {
                for (std::size_t j = 0; j < Ne; ++j) {
                    K(elem_dofs[i], elem_dofs[j]) += Ke(i, j);
                }
            }
        }
    }

    // Apply per-patch conditions (loads, penalty/Nitsche/Lagrange BCs, etc.).
    // Conditions receive the full primal_blocks vector and use their stored
    // patch_idx_ (or interface info, for coupling) to pick the relevant blocks.
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        for (const auto& cond : conditions_per_patch_[p]) {
            cond->apply(K, F, layout_, primal_blocks);
        }
    }

    // Apply exact constraints (Master-Slave first, Dirichlet last).
    for (const auto& constraint : constraints_) {
        constraint->apply(K, F);
    }
    for (const auto& constraint : direct_constraints_) {
        constraint->apply(K, F);
    }
}

// === Template Instantiations ========================================================

template class LinearElasticProblem<double, 1>;
template class LinearElasticProblem<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LinearElasticProblem<float, 1>;
template class LinearElasticProblem<float, 2>;
#endif

} // namespace pyck
