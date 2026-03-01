#include "load_condition.hpp"
#include "../basis/bspline.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
LoadCondition<T, d>::LoadCondition(const Patch<T, d>& patch,
                                   const QuadratureRule<T, d>& quadrature,
                                   const std::function<T(const ColMatrix<T, d>&)>& load_fn)
{
    const auto& points = quadrature.points();
    const auto& weights = quadrature.weights();
    // Initialize the element load vector with zeros
    std::size_t num_basis = patch.dof_mapper().num_basis()[0]; 
    for (std::size_t i = 1; i < d; ++i) {
        num_basis *= patch.dof_mapper().num_basis()[i];
    }
    
    element_load_.setZero(num_basis);

    std::size_t Q = quadrature.num_points();
    
    // Determine number of intervals per dimension
    std::array<std::size_t, d> intervals;
    std::size_t total_elements = 1;

    for (std::size_t i = 0; i < d; ++i) {
        const auto& bspline = dynamic_cast<const BSpline<T>&>(patch.basis(i));
        std::size_t count = bspline.knots().size() - 1;
        intervals[i] = count;
        total_elements *= count;
    }

    // Single linear loop over all possible multidimensional element indices
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

        // 1. Check if the element has non-zero volume in the parametric domain
        std::array<T, d> u_a, u_b;
        bool zero_volume = false;

        for (std::size_t i = 0; i < d; ++i) {
            const auto& bspline = dynamic_cast<const BSpline<T>&>(patch.basis(i));
            const auto& knots = bspline.knots();
            u_a[i] = knots[span_indices[i]];
            u_b[i] = knots[span_indices[i] + 1];

            if (std::abs(u_b[i] - u_a[i]) < 1e-14) {
                zero_volume = true;
                break;
            }
        }

        if (zero_volume) continue;

        // 2. Map quadrature points from [-1, 1]^d to the parametric element box
        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(u_a, u_b);
        
        // 3. Evaluate basis functions and Jacobian
        auto shape_funcs = patch.eval_shape_functions(mapped_pts, 0)[0];
        auto jacobian    = patch.eval_jacobian(mapped_pts);
        
        // 4. Evaluate the traction/load function at quadrature points
        Vector<T> t_vals(Q);
        for (std::size_t q = 0; q < Q; ++q) {
            t_vals(q) = load_fn(mapped_pts.row(q));
        }

        // 5. Integrate: f_ext += \sum_{q} N^T * t * |J| * w_q
        Vector<T> W_J_T = mapped_weights.cwiseProduct(jacobian).cwiseProduct(t_vals);
        element_load_ += shape_funcs.transpose() * W_J_T;
    }

    // Map to global DOFs
    std::size_t K = element_load_.size();
    global_dofs_.reserve(K);
    const auto& mapper = patch.dof_mapper();
    
    std::array<Index, d> current_idx;
    current_idx.fill(0);

    for (std::size_t k = 0; k < K; ++k) 
    {
        global_dofs_.push_back(mapper.to_global(current_idx));
        if (k < K - 1) {
            mapper.next_logical_index(current_idx);
        }
    }
}

template <std::floating_point T, std::size_t d>
void LoadCondition<T, d>::apply(Matrix<T>& /*stiffness*/, Vector<T>& load) const
{
    // The load condition is a RHS-only contribution.
    for (std::size_t k = 0; k < global_dofs_.size(); ++k) {
        Index dof = global_dofs_[k];
        load(dof) += element_load_(k);
    }
}

// === Template Instantiations ========================================================

template class LoadCondition<double, 1>;
template class LoadCondition<double, 2>;
template class LoadCondition<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LoadCondition<float, 1>;
template class LoadCondition<float, 2>;
template class LoadCondition<float, 3>;
#endif

} // namespace pyck