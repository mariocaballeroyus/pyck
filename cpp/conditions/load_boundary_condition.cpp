#include "load_boundary_condition.hpp"

#include "boundary_element_values.hpp"

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
    return boundary_.tensor_product().num_elements()
         * static_cast<Index>(quadrature_.num_points());
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
    const Index ndof = static_cast<Index>(element_.num_node_dofs());
    const Index Q = static_cast<Index>(quadrature_.num_points());

    Index    parent_basis_order = element_.basis_order();
    unsigned parent_flags       = element_.flags();
    for (const auto& term : terms_) {
        parent_basis_order = std::max(parent_basis_order, term.field->basis_order());
        parent_flags |= term.field->flags();
    }
    BoundaryElementValues<T, d> bvals(boundary_, parent_basis_order,
                                      parent_flags, quadrature_);
    const Index num_spans = bvals.num_elements();

    std::size_t qpt_offset = 0;
    for (Index span = 0; span < num_spans; ++span)
    {
        bvals.reinit(span);

        const Index n_elem = static_cast<Index>(bvals.parent_vals_.elem_cps_.size());
        const Index K_elem = n_elem * ndof;

        Vector<T> F_local = Vector<T>::Zero(K_elem);

        for (const auto& term : terms_)
        {
            Matrix<T> C = term.field->evaluate(element_, bvals);

            for (Index q = 0; q < Q; ++q)
            {
                const T t_bar = term.varying
                    ? term.values_at_qpts(static_cast<Eigen::Index>(qpt_offset + q))
                    : term.constant_value;
                const T dGamma =
                    bvals.boundary_vals_.jac(q) * bvals.boundary_vals_.mapped_weights_(q);
                F_local.noalias() += (dGamma * t_bar) * C.row(q).transpose();
            }
        }
        qpt_offset += static_cast<std::size_t>(Q);

        // Scatter F_local into the global load vector.
        layout.scatter_primal(primal_block, bvals.parent_vals_.elem_cps_,
                              bvals.parent_vals_.elem_dofs_);
        const Index n = static_cast<Index>(bvals.parent_vals_.elem_dofs_.size());
        for (Index i = 0; i < n; ++i) {
            load(bvals.parent_vals_.elem_dofs_[i]) += F_local(i);
        }
    }
}

template class LoadBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LoadBoundaryCondition<float, 2>;
#endif

} // namespace pyck
