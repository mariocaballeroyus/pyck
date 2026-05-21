#include "load_boundary_condition.hpp"

#include "patch_boundary.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"

#include <cmath>
#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
requires (d > 1)
LoadBoundaryCondition<T, d>::LoadBoundaryCondition(
    const PatchBoundary<T, d>& boundary,
    const Element<T, d>& element,
    const QuadratureRule<T, d - 1>& quadrature)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature)
{}

template <std::floating_point T, std::size_t d>
requires (d > 1)
LoadBoundaryCondition<T, d>& LoadBoundaryCondition<T, d>::add(
    Ptr<const BoundaryField<T>> field, T value)
{
    if (!field) {
        throw std::invalid_argument(
            "LoadBoundaryCondition::add: field must not be null.");
    }
    Term term;
    term.field = std::move(field);
    term.varying = false;
    term.constant_value = value;
    terms_.push_back(std::move(term));
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
LoadBoundaryCondition<T, d>& LoadBoundaryCondition<T, d>::add(
    Ptr<const BoundaryField<T>> field, const Vector<T>& values_at_qpts)
{
    if (!field) {
        throw std::invalid_argument(
            "LoadBoundaryCondition::add: field must not be null.");
    }
    const Index nq = num_active_qpts();
    if (values_at_qpts.size() != static_cast<Eigen::Index>(nq)) {
        throw std::runtime_error(
            "LoadBoundaryCondition::add: values_at_qpts has size " +
            std::to_string(values_at_qpts.size()) +
            " but expected " + std::to_string(nq) +
            " (one value per active boundary quadrature point).");
    }
    Term term;
    term.field = std::move(field);
    term.varying = true;
    term.values_at_qpts = values_at_qpts;
    terms_.push_back(std::move(term));
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
Index LoadBoundaryCondition<T, d>::num_active_qpts() const
{
    const Index num_spans = boundary_.basis(0).knot_vector().num_spans();
    const std::size_t Q = quadrature_.num_points();
    std::size_t active = 0;
    for (Index s = 0; s < num_spans; ++s) {
        auto [lo, hi] = boundary_.basis(0).knot_vector().span_bounds(s);
        if (std::abs(hi - lo) < T(1e-14)) continue;
        active += Q;
    }
    return static_cast<Index>(active);
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void LoadBoundaryCondition<T, d>::apply(
    Matrix<T>& /*stiffness*/,
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
    const Index Q = static_cast<Index>(quadrature.num_points());
    ColMatrix<T, 1> mapped_pts(Q, 1);
    Vector<T>       mapped_weights(Q);
    std::size_t qpt_offset = 0;
    for (Index span = 0; span < num_spans; ++span)
    {
        // Skip zero-length spans (consistent with active-qpt accounting).
        auto [lo, hi] = boundary.basis(0).knot_vector().span_bounds(span);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        // Per-span scaffolding: shared across all fields.
        quadrature.map_to_domain({lo}, {hi}, mapped_pts, mapped_weights);

        auto boundary_basis  = boundary.tensor_product().eval(mapped_pts, 2);
        auto boundary_act    = boundary.active_control_pts(span);
        IntrinsicGeometry<T, d - 1> boundary_local(boundary_basis, boundary_act);

        const Index flat_parent = boundary.parent_flat_span(span);
        const ColMatrix<T, 2> parent_pts = boundary.lift_to_parent(mapped_pts);
        auto parent_basis  = parent.tensor_product().eval(parent_pts, req_order);
        auto parent_act    = parent.active_control_pts(flat_parent);
        IntrinsicGeometry<T, d> parent_ig(parent_basis, parent_act);
        parent_ig.compute_christoffels();

        std::vector<Index> elem_dofs;
        parent.dof_mapper().get_element_dofs(flat_parent, elem_dofs);
        const Index n_elem = static_cast<Index>(elem_dofs.size());
        const Index K_elem = n_elem * ndof;

        Vector<T> F_local = Vector<T>::Zero(K_elem);

        for (const auto& term : terms_)
        {
            Matrix<T> C = term.field->evaluate(
                element, boundary, span, boundary_basis, boundary_local,
                flat_parent, parent_basis, parent_ig);

            for (Index q = 0; q < Q; ++q)
            {
                const T t_bar = term.varying
                    ? term.values_at_qpts(static_cast<Eigen::Index>(qpt_offset + q))
                    : term.constant_value;
                const T dGamma = boundary_local.jac(q) * mapped_weights(q);
                F_local.noalias() += (dGamma * t_bar) * C.row(q).transpose();
            }
        }
        qpt_offset += Q;

        // Scatter F_local into the global load vector.
        std::vector<Index> dofs;
        layout.scatter_primal(primal_block, elem_dofs, dofs);
        const Index n = static_cast<Index>(dofs.size());
        for (Index i = 0; i < n; ++i) {
            load(dofs[i]) += F_local(i);
        }
    }
}

template class LoadBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LoadBoundaryCondition<float, 2>;
#endif

} // namespace pyck
