#include "linear_elastic_problem.hpp"
#include "bspline.hpp"
#include "direct_constraint.hpp"
#include "patch.hpp"

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

    // Loop over patches
    for (std::size_t p = 0; p < patches_.size(); ++p)
    {
        const auto& patch = *patches_[p];
        const auto& element = *elements_[p];
        const auto& quadrature = *quadratures_[p];
        const auto& mapper = patch.dof_mapper();
        const DofLayout::BlockId primal_block = primal_blocks[p];

        // Collect number of finite elements in the patch
        std::array<std::size_t, d> intervals;
        std::size_t total_elements = 1;
        const auto& tp_intervals = patch.tensor_product().num_intervals();
        for (std::size_t i = 0; i < d; ++i) {
            intervals[i] = tp_intervals[i];
            total_elements *= tp_intervals[i];
        }

        // === Phase 1: enumerate live (non-zero-volume) elements and gather
        //              all quadrature points into one big buffer so the basis
        //              can be evaluated for the whole patch in one call.
        struct LiveElem {
            std::size_t elem_idx;
            Index       qp_offset;
            Index       qp_count;
            Vector<T>   weights;
        };
        std::vector<LiveElem>         live;
        std::vector<ColMatrix<T, d>>  per_elem_pts;
        live.reserve(total_elements);
        per_elem_pts.reserve(total_elements);

        Index Q_total = 0;

        // Loop over finite elements
        for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx) {
            std::array<std::size_t, d> span_indices;
            std::size_t temp_idx = elem_idx;
            for (std::size_t i = 0; i < d; ++i) {
                span_indices[i] = temp_idx % intervals[i];
                temp_idx /= intervals[i];
            }

            std::array<T, d> u_a, u_b;
            bool zero_volume = false;
            for (std::size_t i = 0; i < d; ++i) {
                auto [lo, hi] = patch.basis(i).knot_vector().span_bounds(span_indices[i]);
                u_a[i] = lo;
                u_b[i] = hi;
                if (std::abs(hi - lo) < 1e-14) { zero_volume = true; break; }
            }
            if (zero_volume) continue;

            auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(u_a, u_b);
            const Index qp = static_cast<Index>(mapped_pts.rows());
            live.push_back({elem_idx, Q_total, qp, std::move(mapped_weights)});
            per_elem_pts.push_back(std::move(mapped_pts));
            Q_total += qp;
        }

        if (live.empty()) continue;

        ColMatrix<T, d> big_pts(Q_total, static_cast<Index>(d));
        for (std::size_t i = 0; i < live.size(); ++i) {
            big_pts.middleRows(live[i].qp_offset, live[i].qp_count) = per_elem_pts[i];
        }
        per_elem_pts.clear();
        per_elem_pts.shrink_to_fit();

        const Index order = static_cast<Index>(element.min_order());
        auto big_basis = patch.tensor_product().eval(big_pts, order);

        // === Phase 2: per-element stiffness from sliced basis.
        std::vector<Matrix<T>> elem_basis(order + 1);
        for (const auto& info : live)
        {
            for (Index k = 0; k <= order; ++k) {
                elem_basis[k] = big_basis[k].middleCols(info.qp_offset, info.qp_count);
            }

            element.compute_local_stiffness(patch, info.elem_idx,
                                            elem_basis, info.weights, Ke);

            auto elem_nodes = mapper.get_element_dofs(info.elem_idx);
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
