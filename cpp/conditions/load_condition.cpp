#include "load_condition.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "bspline.hpp"
#include <stdexcept>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
LoadCondition<T, d>::LoadCondition(const Patch<T, d>& patch,
                                   const Element<T, d>& element,
                                   const QuadratureRule<T, d>& quadrature,
                                   const Vector<T>& load_values)
    : patch_(patch)
{
    const Index ndof = element.num_node_dofs();

    // Initialize the element load vector with zeros
    element_load_.setZero(static_cast<Index>(patch.num_control_pts()) * ndof);

    const std::size_t Q = quadrature.num_points();

    // The tensor product already excludes zero-volume (degenerate) elements, so
    // the active quadrature points are simply (live elements) × (points/element).
    const std::size_t num_active_qpts =
        static_cast<std::size_t>(patch.tensor_product().num_elements()) * Q;

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
    ElementValues<T, d> ev(patch, element.basis_order(), element.flags(),
                           quadrature);
    const std::size_t num_live = static_cast<std::size_t>(ev.num_elements());

    for (std::size_t e = 0; e < num_live; ++e)
    {
        ev.reinit(e);
        element.displacement_shape_matrix(ev);
        const Matrix<T>& N_w = element.N_w_workspace_;

        Vector<T> W_J_T(Q);
        for (std::size_t k = 0; k < Q; ++k) {
            T scale = ev.mapped_weights_(k) * ev.jac(k);
            W_J_T(k) = broadcasted_values((qp_offset + k) * ndof) * scale;
        }
        qp_offset += Q;

        // f_local = N_w^T · (q · |J| · w)  — size K_elem
        Vector<T> local_load = N_w.transpose() * W_J_T;

        // Scatter into the element's global DOFs (node-major layout).
        const auto& elem_cps = ev.elem_cps_;
        for (std::size_t k = 0; k < elem_cps.size(); ++k) {
            for (Index v = 0; v < ndof; ++v) {
                element_load_(elem_cps[k] * ndof + v) += local_load(k * ndof + v);
            }
        }
    }

    // Stash CP indices and per-node DOF count; absolute global IDs are
    // resolved at declare_dofs / apply time via the layout.
    cp_indices_ = patch.assembly_dofs();
    node_dofs_ = ndof;
}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
void LoadCondition<T, d>::apply(Matrix<T>& /*stiffness*/,
                                Vector<T>& load,
                                const DofLayout& layout,
                                DofLayout::BlockId primal_block) const
{
    // The load condition is a RHS-only contribution.
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