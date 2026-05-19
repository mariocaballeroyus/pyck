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
    const std::size_t req_order = element_.min_order();

    // Span counts in each parametric direction.
    std::array<std::size_t, d> intervals;
    std::size_t total_elements = 1;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = patch_.basis(i).knot_vector().num_spans();
        total_elements *= intervals[i];
    }

    for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx)
    {
        // Decode flat span index into per-dimension span indices (U-inner).
        std::array<std::size_t, d> span_indices;
        std::size_t temp_idx = elem_idx;
        for (std::size_t i = 0; i < d; ++i) {
            span_indices[i] = temp_idx % intervals[i];
            temp_idx /= intervals[i];
        }

        // Span bounds in parametric domain; skip zero-volume spans.
        std::array<T, d> u_a, u_b;
        bool zero_volume = false;
        for (std::size_t i = 0; i < d; ++i) {
            auto [lo, hi] = patch_.basis(i).knot_vector().span_bounds(span_indices[i]);
            u_a[i] = lo;
            u_b[i] = hi;
            if (std::abs(hi - lo) < T(1e-14)) { zero_volume = true; break; }
        }
        if (zero_volume) continue;

        auto [mapped_pts, mapped_weights] = quadrature_.map_to_domain(u_a, u_b);
        const Index Q = static_cast<Index>(mapped_pts.rows());

        auto basis   = eval_basis(patch_, mapped_pts, elem_idx, req_order);
        auto act_pts = patch_.active_control_pts(elem_idx);
        IntrinsicGeometry local(basis, act_pts);
        const Index n_basis = basis.N();

        auto elem_cps = patch_.dof_mapper().get_element_dofs(elem_idx);
        std::vector<Index> elem_cps_v(elem_cps.begin(), elem_cps.end());
        auto primal_dofs = layout.scatter_primal(primal_block, elem_cps_v);
        // primal_dofs is CP-major, ndof-inner: primal_dofs[i*ndof + v].

        // Element-wise integrated measure dω = w_q · |J|_q, used for the load
        // contribution of every term.
        T elem_measure = T(0);
        for (Index q = 0; q < Q; ++q) {
            elem_measure += mapped_weights(q) * local.jac(q);
        }

        for (const auto& term : terms_)
        {
            const Index slot       = static_cast<Index>(term.dof_index);
            const Index lambda_row = layout.block_base(term.block_id);

            // C_i = ∫ N_i dω over this element. Scattered into the global
            // saddle-point coupling (symmetric).
            Vector<T> C = Vector<T>::Zero(n_basis);
            for (Index q = 0; q < Q; ++q) {
                auto slab0 = basis.data()[0].col(q);
                const T dV = mapped_weights(q) * local.jac(q);
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

            // RHS: g = value · ∫ dω accumulated over elements yields
            // value · |Ω| at the end of assembly.
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
