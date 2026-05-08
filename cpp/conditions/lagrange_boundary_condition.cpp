#include "lagrange_boundary_condition.hpp"

#include "patch_boundary.hpp"
#include "patch.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace pyck
{

template <std::floating_point T, std::size_t d>
requires (d > 1)
LagrangeBoundaryCondition<T, d>::LagrangeBoundaryCondition(
    const PatchBoundary<T, d>& boundary,
    const Element<T, d>& element,
    const QuadratureRule<T, d - 1>& quadrature)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature),
      multiplier_dof_count_(static_cast<Index>(boundary.num_control_pts()))
{
    if (multiplier_dof_count_ == 0) {
        throw std::invalid_argument(
            "LagrangeBoundaryCondition: "
            "boundary has no active basis functions."
        );
    }
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
LagrangeBoundaryCondition<T, d>& LagrangeBoundaryCondition<T, d>::add(
    Ptr<const BoundaryField<T>> field, T value)
{
    if (!field) {
        throw std::invalid_argument(
            "LagrangeBoundaryCondition::add: field must not be null.");
    }
    terms_.push_back({std::move(field), value, 0});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
std::size_t LagrangeBoundaryCondition<T, d>::num_dofs() const
{
    return static_cast<std::size_t>(multiplier_dof_count_) * terms_.size();
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void LagrangeBoundaryCondition<T, d>::allocate_dofs(
    DofLayout& layout,
    const std::vector<DofLayout::BlockId>& primal_blocks)
{
    (void)primal_blocks;
    for (auto& term : terms_) {
        term.block_id = layout.allocate(
            DofType::LagrangeMultiplier, multiplier_dof_count_, 1);
    }
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void LagrangeBoundaryCondition<T, d>::apply(
    Matrix<T>& stiffness, Vector<T>& load, const DofLayout& layout,
    const std::vector<DofLayout::BlockId>& primal_blocks) const
{
    if (terms_.empty()) return;
    const DofLayout::BlockId primal_block = primal_blocks.at(this->patch_idx_);

    const auto& boundary = boundary_;
    const auto& element = element_;
    const auto& quadrature = quadrature_;
    const auto& parent = *boundary.parent();
    const Index ndof = static_cast<Index>(element.num_node_dofs());
    const std::size_t req_order = element.min_order();
    const Index num_spans_bdy = boundary.basis(0).knot_vector().num_spans();

    for (Index s = 0; s < num_spans_bdy; ++s)
    {
        auto [lo, hi] = boundary.basis(0).knot_vector().span_bounds(s);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        // Per-span scaffolding: shared across all fields.
        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        const Index Q = static_cast<Index>(mapped_pts.rows());

        auto [boundary_derivs, boundary_jac] =
            boundary.eval_shape_functions(mapped_pts, s, 1);

        auto multiplier_basis_ids = boundary.dof_mapper().get_element_dofs(s);

        const Index flat_parent = boundary.parent_flat_span(s);
        const ColMatrix<T, 2> parent_pts = boundary.lift_to_parent(mapped_pts);
        auto [parent_derivs, _] =
            parent.eval_shape_functions(parent_pts, flat_parent, req_order);

        auto elem_dofs = parent.dof_mapper().get_element_dofs(flat_parent);
        const Index n_elem = static_cast<Index>(elem_dofs.size());
        const Index K_elem = n_elem * ndof;

        std::vector<Index> elem_node_cps(elem_dofs.begin(), elem_dofs.end());
        auto primal_dofs = layout.scatter_primal(primal_block, elem_node_cps);
        const Index n_primal = static_cast<Index>(primal_dofs.size());
        const Index n_lambda = static_cast<Index>(multiplier_basis_ids.size());

        // Each field assembles its own coupling block but reuses all the
        // shape data computed above.
        for (const auto& term : terms_)
        {
            Matrix<T> C = term. ->evaluate(
                element, boundary, s, boundary_derivs,
                flat_parent, parent_derivs);

            Matrix<T> C_local = Matrix<T>::Zero(n_lambda, K_elem);
            Vector<T> G_local = Vector<T>::Zero(n_lambda);

            for (Index q = 0; q < Q; ++q)
            {
                const T dGamma = boundary_jac(q) * mapped_weights(q);
                C_local.noalias() += dGamma
                    * boundary_derivs[0].row(q).transpose() * C.row(q);
                G_local.noalias() += dGamma * term.value
                    * boundary_derivs[0].row(q).transpose();
            }

            const Index multiplier_base = layout.block_base(term.block_id);
            for (Index i = 0; i < n_lambda; ++i) {
                const Index lambda_i = multiplier_base + multiplier_basis_ids[i];
                load(lambda_i) += G_local(i);

                for (Index j = 0; j < n_primal; ++j) {
                    const Index primal_j = primal_dofs[j];
                    const T cij = C_local(i, j);
                    stiffness(lambda_i, primal_j) += cij;
                    stiffness(primal_j, lambda_i) += cij;
                }
            }
        }
    }
}

template class LagrangeBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LagrangeBoundaryCondition<float, 2>;
#endif

} // namespace pyck
