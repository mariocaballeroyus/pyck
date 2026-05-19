#include "load_condition.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "bspline.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
LoadCondition<T, d>::LoadCondition(const Patch<T, d>& patch,
                                   const Element<T, d>& element,
                                   const QuadratureRule<T, d>& quadrature,
                                   const Vector<T>& load_values)
{
    const Index ndof = element.num_node_dofs();
    Index num_nodes = patch.dof_mapper().num_basis()[0];
    for (Index i = 1; i < d; ++i) {
        num_nodes *= patch.dof_mapper().num_basis()[i];
    }

    std::array<std::size_t, d> intervals;
    std::size_t total_elements = 1;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = patch.basis(i).knot_vector().num_spans();
        total_elements *= intervals[i];
    }

    // Initialize the element load vector with zeros
    element_load_.setZero(num_nodes * ndof);

    const std::size_t Q = quadrature.num_points();

    // Count actual active quadrature points (excluding zero-volume elements)
    std::size_t num_active_qpts = 0;
    for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx) {
        std::array<std::size_t, d> span_indices;
        std::size_t temp_idx = elem_idx;
        for (std::size_t i = 0; i < d; ++i) {
            span_indices[i] = temp_idx % intervals[i];
            temp_idx /= intervals[i];
        }

        bool zero_volume = false;
        for (std::size_t i = 0; i < d; ++i) {
            auto [lo, hi] = patch.basis(i).knot_vector().span_bounds(span_indices[i]);
            if (std::abs(hi - lo) < static_cast<T>(1e-14)) {
                zero_volume = true;
                break;
            }
        }
        if (!zero_volume) {
            num_active_qpts += Q;
        }
    }

    // Handle broadcasting: if load_values has size num_active_qpts, broadcast to first DOF
    Vector<T> broadcasted_values;
    if (load_values.size() == static_cast<Index>(num_active_qpts * ndof)) {
        // Already correct size - use as-is
        broadcasted_values = load_values;
    } else if (load_values.size() == static_cast<Index>(num_active_qpts)) {
        // Scalar load per quadrature point - broadcast to first DOF
        broadcasted_values.resize(num_active_qpts * ndof);
        broadcasted_values.setZero();
        for (std::size_t i = 0; i < num_active_qpts; ++i) {
            broadcasted_values(i * ndof) = load_values(i);
        }
    } else {
        throw std::runtime_error(
            "LoadCondition: load_values has size " + std::to_string(load_values.size()) +
            " but expected either " + std::to_string(num_active_qpts) +
            " (scalar per point) or " + std::to_string(num_active_qpts * ndof) + " (full DOF array)");
    }

    // Track quadrature point offset into broadcasted_values
    std::size_t qp_offset = 0;

    // Single linear loop over all possible multidimensional element indices
    for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx) 
    {
        // Decode linear index into multidimensional span_indices
        std::array<std::size_t, d> span_indices;
        std::size_t temp_idx = elem_idx;
        for (std::size_t i = 0; i < d; ++i)
        {
            span_indices[i] = temp_idx % intervals[i];
            temp_idx /= intervals[i];
        }

        // 1. Check if the element has non-zero volume in the parametric domain
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

        // Simplify: Map quadrature points using the rule's built-in multi-dimensional logic
        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(u_a, u_b);
        
        // Composable geometric primitives.
        std::size_t req_order = element.min_order();
        auto basis   = eval_basis(patch, mapped_pts, elem_idx, req_order);
        auto act_pts = patch.active_control_pts(elem_idx);
        IntrinsicGeometry ig(basis, act_pts);
        ig.compute_christoffels();
        Matrix<T> N_w = element.displacement_shape_matrix(patch, basis, ig);

        Vector<T> W_J_T(Q);
        for (std::size_t k = 0; k < Q; ++k) {
            T scale = mapped_weights(k) * ig.jac(k);
            W_J_T(k) = broadcasted_values((qp_offset + k) * ndof) * scale;
        }
        qp_offset += Q;

        // f_local = N_w^T · (q · |J| · w)  — size K_elem
        Vector<T> local_load = N_w.transpose() * W_J_T;

        // Scatter into the element's global DOFs (node-major layout).
        auto elem_nodes = patch.dof_mapper().get_element_dofs(elem_idx);
        for (std::size_t k = 0; k < elem_nodes.size(); ++k) {
            for (Index v = 0; v < ndof; ++v) {
                element_load_(elem_nodes[k] * ndof + v) += local_load(k * ndof + v);
            }
        }
    }

    // Stash CP indices and per-node DOF count; absolute global IDs are
    // resolved at declare_dofs / apply time via the layout.
    cp_indices_ = patch.assembly_dofs();
    node_dofs_ = ndof;
}

template <std::floating_point T, std::size_t d>
void LoadCondition<T, d>::apply(Matrix<T>& /*stiffness*/,
                                Vector<T>& load,
                                const DofLayout& layout,
                                const std::vector<DofLayout::BlockId>& primal_blocks) const
{
    // The load condition is a RHS-only contribution.
    const DofLayout::BlockId primal_block = primal_blocks.at(this->patch_idx_);
    const Index ndof = node_dofs_;
    const Index base_offset = layout.block_base(primal_block);
    const Index stride = layout.block_stride(primal_block);
    for (std::size_t k = 0; k < cp_indices_.size(); ++k) {
        const Index base = base_offset + cp_indices_[k] * stride;
        for (Index v = 0; v < ndof; ++v) {
            load(base + v) += element_load_(k * ndof + v);
        }
    }
}

// === Template Instantiations ========================================================

template class LoadCondition<double, 1>;
template class LoadCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LoadCondition<float, 1>;
template class LoadCondition<float, 2>;
#endif

} // namespace pyck