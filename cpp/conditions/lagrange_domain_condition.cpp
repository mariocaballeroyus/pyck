#include "lagrange_domain_condition.hpp"

#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pyck
{

template <std::floating_point T, std::size_t d>
LagrangeDomainCondition<T, d>::LagrangeDomainCondition(
    const Patch<T, d>& patch,
    const Element<T, d>& element,
    const QuadratureRule<T, d>& quadrature)
    : patch_(patch),
      element_(element),
      quadrature_(quadrature)
{
}

template <std::floating_point T, std::size_t d>
LagrangeDomainCondition<T, d>& LagrangeDomainCondition<T, d>::add(
    std::size_t dof_index, T value)
{
    if (dof_index >= element_.num_node_dofs()) {
        throw std::invalid_argument(
            "LagrangeDomainCondition::add: dof_index out of range "
            "for the bound element.");
    }
    terms_.push_back({dof_index, value, 0});
    return *this;
}

template <std::floating_point T, std::size_t d>
void LagrangeDomainCondition<T, d>::allocate_dofs(
    DofLayout& layout,
    const std::vector<DofLayout::BlockId>& primal_blocks)
{
    (void)primal_blocks;
    for (auto& term : terms_) {
        // One scalar λ per term.
        term.block_id = layout.allocate(
            DofType::LagrangeMultiplier, /*count=*/1, /*stride=*/1);
    }
}

template <std::floating_point T, std::size_t d>
void LagrangeDomainCondition<T, d>::apply(
    Matrix<T>& stiffness, Vector<T>& load, const DofLayout& layout,
    const std::vector<DofLayout::BlockId>& primal_blocks) const
{
    if (terms_.empty()) return;

    const DofLayout::BlockId primal_block = primal_blocks.at(this->patch_idx_);
    const Index ndof = static_cast<Index>(element_.num_node_dofs());
    const Index Q    = static_cast<Index>(quadrature_.num_points());

    ElementValues<T, d> ev(patch_, Index(1), Flags::Metric, quadrature_);
    const std::size_t num_live = static_cast<std::size_t>(ev.num_elements());

    for (std::size_t e = 0; e < num_live; ++e)
    {
        ev.reinit(e);

        const Index n_basis = ev.results_[0].rows();

        std::vector<Index> primal_dofs;
        layout.scatter_primal(primal_block, ev.elem_cps_, primal_dofs);
        // primal_dofs is CP-major, ndof-inner: primal_dofs[i*ndof + v].

        T elem_measure = T(0);
        for (Index q = 0; q < Q; ++q) {
            elem_measure += ev.mapped_weights_(q) * ev.jac(q);
        }

        for (const auto& term : terms_)
        {
            const Index slot       = static_cast<Index>(term.dof_index);
            const Index lambda_row = layout.block_base(term.block_id);

            Vector<T> C = Vector<T>::Zero(n_basis);
            for (Index q = 0; q < Q; ++q) {
                auto slab0 = ev.results_[0].col(q);
                const T dV = ev.mapped_weights_(q) * ev.jac(q);
                for (Index i = 0; i < n_basis; ++i) {
                    C(i) += dV * slab0(i);
                }
            }
            for (Index i = 0; i < n_basis; ++i) {
                const T C_i = C(i);
                if (C_i == T(0)) continue;
                const Index col = primal_dofs[i * ndof + slot];
                stiffness(lambda_row, col) += C_i;
                stiffness(col, lambda_row) += C_i;
            }

            if (term.value != T(0)) {
                load(lambda_row) += term.value * elem_measure;
            }
        }
    }
}

template class LagrangeDomainCondition<double, 1>;
template class LagrangeDomainCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LagrangeDomainCondition<float, 1>;
template class LagrangeDomainCondition<float, 2>;
#endif

} // namespace pyck
