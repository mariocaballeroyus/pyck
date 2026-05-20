#include "lagrange_coupling_condition.hpp"

#include "patch_boundary.hpp"
#include "patch.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace pyck
{

template <std::floating_point T, std::size_t d>
requires (d > 1)
LagrangeCouplingCondition<T, d>::LagrangeCouplingCondition(
    const PatchBoundary<T, d>& side_a,
    const PatchBoundary<T, d>& side_b,
    std::size_t patch_a_idx,
    std::size_t patch_b_idx,
    const Element<T, d>& element_a,
    const Element<T, d>& element_b,
    const QuadratureRule<T, d - 1>& quadrature,
    bool reverse)
    : side_a_(side_a),
      side_b_(side_b),
      patch_a_idx_(patch_a_idx),
      patch_b_idx_(patch_b_idx),
      element_a_(element_a),
      element_b_(element_b),
      quadrature_(quadrature),
      reverse_(reverse),
      multiplier_dof_count_(static_cast<Index>(side_a.num_control_pts()))
{
    if (patch_a_idx == patch_b_idx) {
        throw std::invalid_argument(
            "LagrangeCouplingCondition: patch_a_idx and patch_b_idx must differ.");
    }
    if (multiplier_dof_count_ == 0) {
        throw std::invalid_argument(
            "LagrangeCouplingCondition: side A boundary has no active basis functions.");
    }
}

// === Utility Methods ================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
LagrangeCouplingCondition<T, d>& LagrangeCouplingCondition<T, d>::add(
    Ptr<const BoundaryField<T>> field_a,
    Ptr<const BoundaryField<T>> field_b,
    T sign_b, T value)
{
    if (!field_a || !field_b) {
        throw std::invalid_argument(
            "LagrangeCouplingCondition::add: field_a and field_b must not be null.");
    }
    terms_.push_back({std::move(field_a), std::move(field_b),
                      sign_b, value, 0});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void LagrangeCouplingCondition<T, d>::allocate_dofs(
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
void 
LagrangeCouplingCondition<T, d>::apply(Matrix<T>& stiffness,
                                       Vector<T>& load,
                                       const DofLayout& layout,
                                       const std::vector<DofLayout::BlockId>& primal_blocks) const
{
    if (terms_.empty()) return;

    const DofLayout::BlockId block_a = primal_blocks.at(patch_a_idx_);
    const DofLayout::BlockId block_b = primal_blocks.at(patch_b_idx_);

    const Patch<T, 2>& parent_a = *side_a_.parent();
    const Patch<T, 2>& parent_b = *side_b_.parent();
    const std::size_t req_order_a = element_a_.min_order();
    const std::size_t req_order_b = element_b_.min_order();

    const auto& kv_a = side_a_.basis(0).knot_vector();
    const Index num_spans = kv_a.num_spans();

    // Domain-flip constant for reverse mapping (palindromic knot symmetry).
    const auto& kv_b = side_b_.basis(0).knot_vector();
    const T b_total = kv_b[0] + kv_b[kv_b.size() - 1];

    for (Index span_a = 0; span_a < num_spans; ++span_a)
    {
        auto [lo, hi] = kv_a.span_bounds(span_a);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        // Quadrature on side A's span.
        auto [pts_a, w_a] = quadrature_.map_to_domain(lo, hi);
        const Index Q = static_cast<Index>(pts_a.rows());

        // Side A: composable primitives on boundary and parent.
        auto basis_a   = side_a_.tensor_product().eval(pts_a, 2);
        auto act_a     = side_a_.active_control_pts(span_a);
        IntrinsicGeometry<T, d - 1> local_a(basis_a, act_a);
        const Index flat_a = side_a_.parent_flat_span(span_a);
        const ColMatrix<T, 2> par_pts_a = side_a_.lift_to_parent(pts_a);
        auto parent_basis_a = parent_a.tensor_product().eval(par_pts_a, req_order_a);
        auto parent_act_a   = parent_a.active_control_pts(flat_a);
        IntrinsicGeometry<T, d> parent_ig_a(parent_basis_a, parent_act_a);
        parent_ig_a.compute_christoffels();

        // Side B: corresponding span and parametric points.
        const Index span_b = reverse_ ? (num_spans - 1 - span_a) : span_a;
        Vector<T> pts_b(Q);
        if (reverse_) {
            for (Index q = 0; q < Q; ++q) pts_b(q) = b_total - pts_a(q);
        } else {
            pts_b = pts_a;
        }

        auto basis_b   = side_b_.tensor_product().eval(pts_b, 2);
        auto act_b     = side_b_.active_control_pts(span_b);
        IntrinsicGeometry<T, d - 1> local_b(basis_b, act_b);
        // local_a.jac is the surface measure dΓ; local_b.jac agrees up to round-off.
        const Index flat_b = side_b_.parent_flat_span(span_b);
        const ColMatrix<T, 2> par_pts_b = side_b_.lift_to_parent(pts_b);
        auto parent_basis_b = parent_b.tensor_product().eval(par_pts_b, req_order_b);
        auto parent_act_b   = parent_b.active_control_pts(flat_b);
        IntrinsicGeometry<T, d> parent_ig_b(parent_basis_b, parent_act_b);
        parent_ig_b.compute_christoffels();

        // Primal DOFs on each side.
        auto cps_a = parent_a.dof_mapper().get_element_dofs(flat_a);
        auto cps_b = parent_b.dof_mapper().get_element_dofs(flat_b);
        std::vector<Index> cps_a_v(cps_a.begin(), cps_a.end());
        std::vector<Index> cps_b_v(cps_b.begin(), cps_b.end());
        auto dofs_a = layout.scatter_primal(block_a, cps_a_v);
        auto dofs_b = layout.scatter_primal(block_b, cps_b_v);
        const Index Na = static_cast<Index>(dofs_a.size());
        const Index Nb = static_cast<Index>(dofs_b.size());

        // Multiplier basis lives on side A's boundary basis.
        auto mult_ids = side_a_.dof_mapper().get_element_dofs(span_a);
        const Index Nl = static_cast<Index>(mult_ids.size());

        for (const auto& term : terms_)
        {
            Matrix<T> C_a = term.field_a->evaluate(
                element_a_, side_a_, span_a, basis_a, local_a,
                flat_a, parent_basis_a, parent_ig_a);
            Matrix<T> C_b = term.field_b->evaluate(
                element_b_, side_b_, span_b, basis_b, local_b,
                flat_b, parent_basis_b, parent_ig_b);

            Matrix<T> Cla = Matrix<T>::Zero(Nl, Na);
            Matrix<T> Clb = Matrix<T>::Zero(Nl, Nb);
            Vector<T> G   = Vector<T>::Zero(Nl);

            for (Index q = 0; q < Q; ++q)
            {
                const T dG = local_a.jac(q) * w_a(q);
                auto slab0_a = basis_a.data()[0].col(q);
                Cla.noalias() += dG * slab0_a * C_a.row(q);
                Clb.noalias() += dG * slab0_a * C_b.row(q);
                if (term.value != T(0)) {
                    G.noalias() += (dG * term.value) * slab0_a;
                }
            }

            const Index mb = layout.block_base(term.block_id);
            const T sB = term.sign_b;
            for (Index i = 0; i < Nl; ++i) {
                const Index lam_i = mb + mult_ids[i];
                load(lam_i) += G(i);
                for (Index j = 0; j < Na; ++j) {
                    const T v = Cla(i, j);
                    stiffness(lam_i, dofs_a[j]) += v;
                    stiffness(dofs_a[j], lam_i) += v;
                }
                for (Index j = 0; j < Nb; ++j) {
                    const T v = sB * Clb(i, j);
                    stiffness(lam_i, dofs_b[j]) -= v;
                    stiffness(dofs_b[j], lam_i) -= v;
                }
            }
        }
    }
}

// === Template Instantiations ========================================================

template class LagrangeCouplingCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LagrangeCouplingCondition<float, 2>;
#endif

} // namespace pyck
