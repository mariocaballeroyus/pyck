#include "linear_elastic_problem.hpp"
#include "../basis/bspline.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
LinearElasticProblem<T, d>::LinearElasticProblem(const Ptr<Patch<T, d>>& patch,
                                                 const Ptr<Element<T, d>>& element)
    : patch_(patch), element_(element), quadrature_(nullptr)
{
}

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::set_quadrature(const Ptr<QuadratureRule<T>>& quadrature)
{
    quadrature_ = quadrature;
}

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::add_condition(const Ptr<Condition<T>>& condition)
{
    conditions_.push_back(condition);
}

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::add_constraint(const Ptr<Constraint<T>>& constraint)
{
    constraints_.push_back(constraint);
}

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
    std::size_t global_dofs = 1;
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

        // Create mapped quadrature rule points and weights for the element
        std::array<const QuadratureRule<T>*, d> q_rules;
        q_rules.fill(quadrature_.get());
        auto [mapped_pts, mapped_weights] = tensor_product_mapped<T, d>(q_rules, u_a, u_b);
        
        // Gather element contributions (span-local stiffness)
        element_->compute_local_stiffness(*patch_, mapped_pts, mapped_weights, elem_idx, Ke);

        // Scatter local stiffness into global matrix
        auto elem_dofs = mapper.get_element_dofs(elem_idx);
        for (std::size_t i = 0; i < elem_dofs.size(); ++i) {
            for (std::size_t j = 0; j < elem_dofs.size(); ++j) {
                K(elem_dofs[i], elem_dofs[j]) += Ke(i, j);
            }
        }
    }

    // Apply boundary / load conditions
    for (const auto& cond : conditions_) {
        // Evaluate the condition on K and F.
        cond->apply(K, F);
    }

    // Apply exact constraints (e.g. Dirichlet, Master-Slave)
    for (const auto& constraint : constraints_) {
        constraint->apply(K, F);
    }
}

// === Template Instantiations ========================================================

template class LinearElasticProblem<double, 1>;
template class LinearElasticProblem<double, 2>;
template class LinearElasticProblem<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LinearElasticProblem<float, 1>;
template class LinearElasticProblem<float, 2>;
template class LinearElasticProblem<float, 3>;
#endif

} // namespace pyck
