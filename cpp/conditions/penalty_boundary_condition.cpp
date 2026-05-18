#include "penalty_boundary_condition.hpp"

#include "patch_boundary.hpp"
#include "patch.hpp"
#include "basis_derivs.hpp"
#include "intrinsic_geometry.hpp"

#include <cmath>
#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyBoundaryCondition<T, d>::PenaltyBoundaryCondition(
    const PatchBoundary<T, d>& boundary,
    const Element<T, d>& element,
    const QuadratureRule<T, d - 1>& quadrature)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature)
{}

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyBoundaryCondition<T, d>& PenaltyBoundaryCondition<T, d>::add(
    Ptr<const BoundaryField<T>> field, T penalty, T value)
{
    if (!field) {
        throw std::invalid_argument("PenaltyBoundaryCondition::add: field must not be null.");
    }
    terms_.push_back({std::move(field), penalty, value});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void PenaltyBoundaryCondition<T, d>::apply(Matrix<T>& stiffness,
                                   Vector<T>& load,
                                   const DofLayout& layout,
                                   const std::vector<DofLayout::BlockId>& primal_blocks) const
{
    if (terms_.empty()) return;
    const DofLayout::BlockId primal_block = primal_blocks.at(this->patch_idx_);

    const PatchBoundary<T, 2>& boundary = boundary_;
    const auto& element = element_;
    const auto& quadrature = quadrature_;
    const Patch<T, 2>& parent = *boundary.parent();
    const Index ndof = static_cast<Index>(element.num_node_dofs());
    const std::size_t req_order = element.min_order();

    const Index num_spans = boundary.basis(0).knot_vector().num_spans();
    for (Index span = 0; span < num_spans; ++span)
    {
        // Skip zero-length spans.
        auto [lo, hi] = boundary.basis(0).knot_vector().span_bounds(span);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        // Per-span scaffolding: shared across all fields.
        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        const Index Q = static_cast<Index>(mapped_pts.rows());

        auto boundary_basis  = eval_basis(boundary, mapped_pts, span, 2);
        auto boundary_act    = boundary.active_control_pts(span);
        IntrinsicGeometry boundary_local(boundary_basis, boundary_act);

        const Index flat_parent = boundary.parent_flat_span(span);
        const ColMatrix<T, 2> parent_pts = boundary.lift_to_parent(mapped_pts);
        auto parent_basis  = eval_basis(parent, parent_pts, flat_parent, req_order);
        auto parent_act    = parent.active_control_pts(flat_parent);
        IntrinsicGeometry parent_ig(parent_basis, parent_act);
        parent_ig.compute_christoffels();

        auto elem_dofs = parent.dof_mapper().get_element_dofs(flat_parent);
        const Index n_elem = static_cast<Index>(elem_dofs.size());
        const Index K_elem = n_elem * ndof;

        Matrix<T> K_local = Matrix<T>::Zero(K_elem, K_elem);
        Vector<T> F_local = Vector<T>::Zero(K_elem);

        // Accumulate every field's contribution into the same local matrices.
        for (const auto& term : terms_)
        {
            Matrix<T> C = term.field->evaluate(
                element, boundary, span, boundary_basis, boundary_local,
                flat_parent, parent_basis, parent_ig);

            for (Index q = 0; q < Q; ++q)
            {
                const T dGamma = boundary_local.jac(q) * mapped_weights(q);
                K_local.noalias() += dGamma * term.penalty
                                   * C.row(q).transpose() * C.row(q);
                F_local.noalias() += dGamma * term.penalty * term.value
                                   * C.row(q).transpose();
            }
        }

        // Scatter once per span.
        std::vector<Index> cps(elem_dofs.begin(), elem_dofs.end());
        auto dofs = layout.scatter_primal(primal_block, cps);
        const Index n = static_cast<Index>(dofs.size());
        for (Index i = 0; i < n; ++i)
        {
            const Index gi = dofs[i];
            load(gi) += F_local(i);
            for (Index j = 0; j < n; ++j) {
                stiffness(gi, dofs[j]) += K_local(i, j);
            }
        }
    }
}

template class PenaltyBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PenaltyBoundaryCondition<float, 2>;
#endif

} // namespace pyck
