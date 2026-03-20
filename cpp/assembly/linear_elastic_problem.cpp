#include "linear_elastic_problem.hpp"
#include "bspline.hpp"
#include "direct_constraint.hpp"
#include "patch.hpp"
#include "factories.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::assemble(Matrix<T>& K, Vector<T>& F) const
{
    if (!quadrature_) {
        throw std::runtime_error("LinearElasticProblem::assemble: "
                                 "quadrature rule not set. "
        );
    }

    // Determine the total number of global DOFs
    const auto& mapper = patch_->dof_mapper();
    std::size_t ndof = element_->num_node_dofs();
    std::size_t global_dofs = ndof;
    const auto& num_basis_array = mapper.num_basis();
    for (std::size_t i = 0; i < d; ++i) {
        global_dofs *= num_basis_array[i];
    }

    // Initialize global matrices to zero
    K.setZero(global_dofs, global_dofs);
    F.setZero(global_dofs);

    // Determine the number of intervals (elements) per dimension
    std::array<std::size_t, d> intervals;
    std::size_t total_elements = 1;

    const auto& tp_intervals = patch_->tensor_product().num_intervals();
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = tp_intervals[i];
        total_elements *= tp_intervals[i];
    }

    // Allocate local (element) matrices
    Matrix<T> Ke;
    
    // Iterate over elements
    for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx) 
    {
        // Decode linear index into multidimensional span_indices
        std::array<std::size_t, d> span_indices;
        std::size_t temp_idx = elem_idx;
        for (std::size_t i = d; i-- > 0; ) // Lexicographical decoding
        {
            span_indices[i] = temp_idx % intervals[i];
            temp_idx /= intervals[i];
        }

        // Check if the element has non-zero volume in the parametric domain
        std::array<T, d> u_a, u_b;
        bool zero_volume = false;

        for (std::size_t i = 0; i < d; ++i) {
            auto [lo, hi] = patch_->basis(i).knot_vector().span_bounds(span_indices[i]);
            u_a[i] = lo;
            u_b[i] = hi;

            if (std::abs(hi - lo) < 1e-14) {
                zero_volume = true;
                break;
            }
        }

        if (zero_volume) continue;

        // Simplify: Map quadrature points using the rule's built-in multi-dimensional logic
        auto [mapped_pts, mapped_weights] = quadrature_->map_to_domain(u_a, u_b);
        
        // Pre-compute shape functions and Jacobian once per element
        auto [shape_fns, jac] = patch_->eval_shape_functions(mapped_pts, elem_idx, element_->min_order());

        // Gather element contributions (span-local stiffness)
        element_->compute_local_stiffness(shape_fns, jac, mapped_weights, Ke);

        // Scatter local stiffness into global matrix
        auto elem_nodes = mapper.get_element_dofs(elem_idx);
        for (std::size_t i = 0; i < elem_nodes.size(); ++i) {
            for (std::size_t j = 0; j < elem_nodes.size(); ++j) {
                for (std::size_t di = 0; di < ndof; ++di) {
                    for (std::size_t dj = 0; dj < ndof; ++dj) {
                        K(elem_nodes[i] * ndof + di, elem_nodes[j] * ndof + dj) += Ke(i * ndof + di, j * ndof + dj);
                    }
                }
            }
        }
    }

    // Apply boundary / load conditions
    for (const auto& cond : conditions_) {
        // Evaluate the condition on K and F.
        cond->apply(K, F);
    }

    // Apply exact constraints (Master-Slave FIRST, Dirichlet LAST)
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
