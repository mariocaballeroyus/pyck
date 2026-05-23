#include "penalty_boundary_condition.hpp"

#include "boundary_element_values.hpp"

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
    const Index ndof = static_cast<Index>(element_.num_node_dofs());
    const Index Q = static_cast<Index>(quadrature_.num_points());

    Index    parent_basis_order = element_.basis_order();
    unsigned parent_flags       = Flags::None;
    for (const auto& term : terms_) {
        parent_basis_order = std::max(parent_basis_order, term.field->basis_order());
        parent_flags |= term.field->flags();
        parent_flags |= term.field->element_flags(element_);
    }
    BoundaryElementValues<T, d> bvals(boundary_, parent_basis_order,
                                      parent_flags, quadrature_);
    const Index num_spans = bvals.num_elements();

    for (Index span = 0; span < num_spans; ++span)
    {
        bvals.reinit(span);

        const Index n_elem = static_cast<Index>(bvals.parent_vals_.elem_cps_.size());
        const Index K_elem = n_elem * ndof;

        Matrix<T> K_local = Matrix<T>::Zero(K_elem, K_elem);
        Vector<T> F_local = Vector<T>::Zero(K_elem);

        // Accumulate every field's contribution into the same local matrices.
        for (const auto& term : terms_)
        {
            Matrix<T> C = term.field->evaluate(element_, bvals);

            for (Index q = 0; q < Q; ++q)
            {
                const T dGamma =
                    bvals.boundary_vals_.jac(q) * bvals.boundary_vals_.mapped_weights_(q);
                K_local.noalias() += dGamma * term.penalty
                                   * C.row(q).transpose() * C.row(q);
                F_local.noalias() += dGamma * term.penalty * term.value
                                   * C.row(q).transpose();
            }
        }

        // Scatter once per span.
        layout.scatter_primal(primal_block, bvals.parent_vals_.elem_cps_,
                              bvals.parent_vals_.elem_dofs_);
        const Index n = static_cast<Index>(bvals.parent_vals_.elem_dofs_.size());
        for (Index i = 0; i < n; ++i)
        {
            const Index gi = bvals.parent_vals_.elem_dofs_[i];
            load(gi) += F_local(i);
            for (Index j = 0; j < n; ++j) {
                stiffness(gi, bvals.parent_vals_.elem_dofs_[j]) += K_local(i, j);
            }
        }
    }
}

template class PenaltyBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PenaltyBoundaryCondition<float, 2>;
#endif

} // namespace pyck
