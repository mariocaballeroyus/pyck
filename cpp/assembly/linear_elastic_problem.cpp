#include "linear_elastic_problem.hpp"
#include "../basis/bspline.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
LinearElasticProblem<T, d>::LinearElasticProblem(const Ptr<Patch<T, d>>& patch,
                                                 const Ptr<Element<T, d>>& element,
                                                 const Ptr<QuadratureRule<T, d>>& quadrature)
    : patch_(patch), element_(element), quadrature_(quadrature)
{
}

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::add_condition(const Ptr<Condition<T>>& condition)
{
    conditions_.push_back(condition);
}

template <std::floating_point T, std::size_t d>
void LinearElasticProblem<T, d>::assemble(Matrix<T>& K, Vector<T>& F) const
{
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
            const auto& basis = patch_->basis(i);
            const auto& knots = basis.knots();
            u_a[i] = knots[span_indices[i]];
            u_b[i] = knots[span_indices[i] + 1];

            if (std::abs(u_b[i] - u_a[i]) < 1e-14) {
                zero_volume = true;
                break;
            }
        }

        // Create mapped quadrature rule points and weights for the element
        auto [mapped_pts, mapped_weights] = quadrature_->map_to_domain(u_a, u_b);
        
        // Gather element contributions
        element_->compute_local_stiffness(*patch_, mapped_pts, mapped_weights, Ke);
        K += Ke;
    }

    // Apply boundary / load conditions
    for (const auto& cond : conditions_) {
        // Evaluate the condition on K and F.
        // Dirichlet condition will modify both; Load condition just adds to F.
        cond->apply(K, F);
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
