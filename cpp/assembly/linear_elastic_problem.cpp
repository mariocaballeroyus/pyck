#include "linear_elastic_problem.hpp"
#include "bspline.hpp"
#include "direct_constraint.hpp"
#include "patch.hpp"
#include "element_values.hpp"
#include "system_assembler.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace pyck
{

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::assemble(SparseMatrix<T>& K, Vector<T>& F) const
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

    // Resolver mapping patch identity to its primal block. A single-patch
    // condition resolves its own patch; a coupling condition resolves both.
    PatchBlocks<T, d> blocks;
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        blocks.add(*patches_[p], primal_blocks[p]);
    }

    // Allocate auxiliary DOFs per patch (e.g. Lagrange multipliers).
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        for (const auto& cond : conditions_per_patch_[p]) {
            cond->allocate_dofs(layout_);
        }
    }

    // Estimate the global triplet count up front
    std::size_t nnz_estimate = 0;
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        std::size_t nodes = 1;
        for (std::size_t i = 0; i < d; ++i) {
            nodes *= static_cast<std::size_t>(patches_[p]->basis(i).degree()) + 1;
        }
        const std::size_t elem_dofs = nodes * elements_[p]->num_node_dofs();
        nnz_estimate += static_cast<std::size_t>(patches_[p]->tensor_product().num_elements())
                        * elem_dofs * elem_dofs;
    }

    // Sparse-assembly sink for the global stiffness and load
    const Index total_dofs = static_cast<Index>(layout_.num_dofs());
    SystemAssembler<T> assembler(total_dofs, nnz_estimate);

    // Single allocation of element stiffness and local load
    Matrix<T> K_local;
    Vector<T> f_local;

    // --- Patch Loop -----------------------------------------------------------------
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        const auto& patch = *patches_[p];
        const auto& element = *elements_[p];
        const auto& quadrature = *quadratures_[p];
        const DofLayout::BlockId primal_block = primal_blocks[p];

        ElementValues<T, d> patch_values(patch, element.basis_order(),
                                         element.flags(),
                                         quadrature);
        const std::size_t num_live = static_cast<std::size_t>(patch_values.num_elements());

        // --- Element Loop -----------------------------------------------------------
        for (std::size_t e = 0; e < num_live; ++e) {
            patch_values.reinit(e);

            element.compute_local_stiffness(patch_values, K_local);

            layout_.scatter_primal(primal_block, patch_values.elem_cps_,
                                   patch_values.elem_dofs_);
            const auto& elem_dofs = patch_values.elem_dofs_;
            const std::size_t Ne = elem_dofs.size();

            // Accumulate element stiffness triplets
            for (std::size_t i = 0; i < Ne; ++i) {
                for (std::size_t j = 0; j < Ne; ++j) {
                    assembler.add_stiffness(elem_dofs[i], elem_dofs[j], K_local(i, j));
                }
            }

            // Accumulate element body-force contributions
            for (const auto& load_fn : domain_loads_per_patch_[p]) {
                element.compute_local_domain_load(patch_values, load_fn, f_local);
                for (std::size_t i = 0; i < Ne; ++i) {
                    assembler.add_load(elem_dofs[i], f_local(i));
                }
            }
        }
    }

    // --- Conditions -----------------------------------------------------------------
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        for (const auto& cond : conditions_per_patch_[p]) {
            cond->apply(assembler, layout_, blocks);
        }
    }

    // Build the sparse system from the accumulated triplets + load
    K = assembler.finalize_matrix();
    F = std::move(assembler.load());

    // --- Constraints ----------------------------------------------------------------
    for (const auto& constraint : constraints_) {
        constraint->apply(K, F);
    }
    for (const auto& constraint : direct_constraints_) {
        constraint->apply(K, F);
    }

    K.makeCompressed();
}

// === Template Instantiations ========================================================

template class LinearElasticProblem<double, 1>;
template class LinearElasticProblem<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LinearElasticProblem<float, 1>;
template class LinearElasticProblem<float, 2>;
#endif

} // namespace pyck
