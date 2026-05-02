#include "linear_elastic_problem.hpp"
#include "bspline.hpp"
#include "direct_constraint.hpp"
#include "patch.hpp"
#include "factories.hpp"

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

    // Reset and allocate one primal DOF block per patch.
    layout_.clear();
    std::vector<DofLayout::BlockId> primal_blocks;
    primal_blocks.reserve(patches_.size());

    for (std::size_t p = 0; p < patches_.size(); ++p) {
        if (!quadratures_[p]) {
            throw std::runtime_error(
                "LinearElasticProblem::assemble: quadrature rule not set for patch "
                + std::to_string(p) + ".");
        }
        const std::size_t ndof = elements_[p]->num_node_dofs();
        const std::size_t num_cps = patches_[p]->num_control_pts();
        primal_blocks.push_back(
            layout_.allocate(DofType::Primal, num_cps * ndof, ndof));
    }

    // Conditions allocate their auxiliary DOFs against their owning patch's block.
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        for (const auto& cond : conditions_per_patch_[p]) {
            cond->allocate_dofs(layout_, primal_blocks[p]);
        }
    }

    const std::size_t total_dofs = layout_.num_dofs();
    K.setZero(total_dofs, total_dofs);
    F.setZero(total_dofs);

    Matrix<T> Ke;

    // Per-patch element loop.
    for (std::size_t p = 0; p < patches_.size(); ++p)
    {
        const auto& patch = *patches_[p];
        const auto& element = *elements_[p];
        const auto& quadrature = *quadratures_[p];
        const auto& mapper = patch.dof_mapper();
        const DofLayout::BlockId primal_block = primal_blocks[p];

        std::array<std::size_t, d> intervals;
        std::size_t total_elements = 1;
        const auto& tp_intervals = patch.tensor_product().num_intervals();
        for (std::size_t i = 0; i < d; ++i) {
            intervals[i] = tp_intervals[i];
            total_elements *= tp_intervals[i];
        }

        for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx)
        {
            // Decode linear index into multidimensional span_indices
            std::array<std::size_t, d> span_indices;
            std::size_t temp_idx = elem_idx;
            for (std::size_t i = 0; i < d; ++i) // Lexicographical decoding
            {
                span_indices[i] = temp_idx % intervals[i];
                temp_idx /= intervals[i];
            }

            // Skip elements with zero parametric volume.
            std::array<T, d> u_a, u_b;
            bool zero_volume = false;
            for (std::size_t i = 0; i < d; ++i) {
                auto [lo, hi] = patch.basis(i).knot_vector().span_bounds(span_indices[i]);
                u_a[i] = lo;
                u_b[i] = hi;
                if (std::abs(hi - lo) < 1e-14) {
                    zero_volume = true;
                    break;
                }
            }
            if (zero_volume) continue;

            // Map quadrature points and evaluate shape functions on this span.
            auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(u_a, u_b);
            auto [shape_fns, jac] = patch.eval_shape_functions(
                mapped_pts, elem_idx, element.min_order());

            // Local element stiffness.
            element.compute_local_stiffness(shape_fns, jac, mapped_weights, Ke);

            // Scatter into the per-patch primal block.
            auto elem_nodes = mapper.get_element_dofs(elem_idx);
            auto elem_dofs = layout_.scatter_primal(primal_block, elem_nodes);
            const std::size_t Ne = elem_dofs.size();
            for (std::size_t i = 0; i < Ne; ++i) {
                for (std::size_t j = 0; j < Ne; ++j) {
                    K(elem_dofs[i], elem_dofs[j]) += Ke(i, j);
                }
            }
        }
    }

    // Apply per-patch conditions (loads, penalty/Nitsche/Lagrange BCs, etc.).
    for (std::size_t p = 0; p < patches_.size(); ++p) {
        for (const auto& cond : conditions_per_patch_[p]) {
            cond->apply(K, F, layout_, primal_blocks[p]);
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
