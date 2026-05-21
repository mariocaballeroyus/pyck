#include "penalty_coupling_condition.hpp"

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
PenaltyCouplingCondition<T, d>::PenaltyCouplingCondition(
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
      reverse_(reverse)
{
    if (patch_a_idx == patch_b_idx) {
        throw std::invalid_argument(
            "PenaltyCouplingCondition: patch_a_idx and patch_b_idx must differ.");
    }
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>& PenaltyCouplingCondition<T, d>::add(
    Ptr<const BoundaryField<T>> field_a,
    Ptr<const BoundaryField<T>> field_b,
    T penalty, T sign_b, T value)
{
    if (!field_a || !field_b) {
        throw std::invalid_argument(
            "PenaltyCouplingCondition::add: field_a and field_b must not be null.");
    }
    terms_.push_back({std::move(field_a), std::move(field_b),
                      penalty, sign_b, value});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void PenaltyCouplingCondition<T, d>::apply(
    Matrix<T>& stiffness,
    Vector<T>& load,
    const DofLayout& layout,
    const std::vector<DofLayout::BlockId>& primal_blocks) const
{
    if (terms_.empty()) return;

    const DofLayout::BlockId block_a = primal_blocks.at(patch_a_idx_);
    const DofLayout::BlockId block_b = primal_blocks.at(patch_b_idx_);

    const Patch<T, 2>& parent_a = *side_a_.parent();
    const Patch<T, 2>& parent_b = *side_b_.parent();
    const Index ndof_a = static_cast<Index>(element_a_.num_node_dofs());
    const Index ndof_b = static_cast<Index>(element_b_.num_node_dofs());
    const std::size_t req_order_a = element_a_.min_order();
    const std::size_t req_order_b = element_b_.min_order();

    const auto& kv_a = side_a_.basis(0).knot_vector();
    const Index num_spans = kv_a.num_spans();

    // Domain-flip constant for reverse mapping (palindromic knot symmetry).
    const auto& kv_b = side_b_.basis(0).knot_vector();
    const T b_total = kv_b[0] + kv_b[kv_b.size() - 1];

    const Index Q = static_cast<Index>(quadrature_.num_points());
    ColMatrix<T, 1> pts_a(Q, 1);
    Vector<T>       w_a(Q);
    for (Index span_a = 0; span_a < num_spans; ++span_a)
    {
        auto [lo, hi] = kv_a.span_bounds(span_a);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        // Quadrature on side A's span.
        quadrature_.map_to_domain({lo}, {hi}, pts_a, w_a);

        // Side A: composable primitives on boundary and parent.
        auto basis_a   = side_a_.tensor_product().eval_all(pts_a, 2);
        auto act_a     = side_a_.active_control_pts(span_a);
        IntrinsicGeometry<T, d - 1> local_a(basis_a, act_a);
        const Index flat_a = side_a_.parent_flat_span(span_a);
        const ColMatrix<T, 2> par_pts_a = side_a_.lift_to_parent(pts_a);
        auto parent_basis_a = parent_a.tensor_product().eval_all(par_pts_a, req_order_a);
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

        auto basis_b   = side_b_.tensor_product().eval_all(pts_b, 2);
        auto act_b     = side_b_.active_control_pts(span_b);
        IntrinsicGeometry<T, d - 1> local_b(basis_b, act_b);
        // local_a.jac is the surface measure dΓ; local_b.jac agrees up to round-off.
        const Index flat_b = side_b_.parent_flat_span(span_b);
        const ColMatrix<T, 2> par_pts_b = side_b_.lift_to_parent(pts_b);
        auto parent_basis_b = parent_b.tensor_product().eval_all(par_pts_b, req_order_b);
        auto parent_act_b   = parent_b.active_control_pts(flat_b);
        IntrinsicGeometry<T, d> parent_ig_b(parent_basis_b, parent_act_b);
        parent_ig_b.compute_christoffels();

        // DOFs on each side.
        std::vector<Index> cps_a, cps_b;
        parent_a.dof_mapper().get_element_dofs(flat_a, cps_a);
        parent_b.dof_mapper().get_element_dofs(flat_b, cps_b);
        std::vector<Index> dofs_a, dofs_b;
        layout.scatter_primal(block_a, cps_a, dofs_a);
        layout.scatter_primal(block_b, cps_b, dofs_b);
        const Index Na = static_cast<Index>(dofs_a.size());
        const Index Nb = static_cast<Index>(dofs_b.size());

        // Local 2x2-block stiffness and 2-block load.
        Matrix<T> Kaa = Matrix<T>::Zero(Na, Na);
        Matrix<T> Kab = Matrix<T>::Zero(Na, Nb);
        Matrix<T> Kba = Matrix<T>::Zero(Nb, Na);
        Matrix<T> Kbb = Matrix<T>::Zero(Nb, Nb);
        Vector<T> Fa = Vector<T>::Zero(Na);
        Vector<T> Fb = Vector<T>::Zero(Nb);

        for (const auto& term : terms_)
        {
            Matrix<T> C_a = term.field_a->evaluate(
                element_a_, side_a_, span_a, basis_a, local_a,
                flat_a, parent_basis_a, parent_ig_a);
            Matrix<T> C_b = term.field_b->evaluate(
                element_b_, side_b_, span_b, basis_b, local_b,
                flat_b, parent_basis_b, parent_ig_b);

            for (Index q = 0; q < Q; ++q)
            {
                const T dG = local_a.jac(q) * w_a(q);
                const T a = dG * term.penalty;
                const T sB = term.sign_b;

                Kaa.noalias() += a *      C_a.row(q).transpose() * C_a.row(q);
                Kab.noalias() -= (a * sB) * C_a.row(q).transpose() * C_b.row(q);
                Kba.noalias() -= (a * sB) * C_b.row(q).transpose() * C_a.row(q);
                Kbb.noalias() += a *      C_b.row(q).transpose() * C_b.row(q);

                if (term.value != T(0)) {
                    Fa.noalias() += (a * term.value)     * C_a.row(q).transpose();
                    Fb.noalias() -= (a * sB * term.value) * C_b.row(q).transpose();
                }
            }
        }

        // Scatter into K and F.
        for (Index i = 0; i < Na; ++i) {
            load(dofs_a[i]) += Fa(i);
            for (Index j = 0; j < Na; ++j)
                stiffness(dofs_a[i], dofs_a[j]) += Kaa(i, j);
            for (Index j = 0; j < Nb; ++j)
                stiffness(dofs_a[i], dofs_b[j]) += Kab(i, j);
        }
        for (Index i = 0; i < Nb; ++i) {
            load(dofs_b[i]) += Fb(i);
            for (Index j = 0; j < Na; ++j)
                stiffness(dofs_b[i], dofs_a[j]) += Kba(i, j);
            for (Index j = 0; j < Nb; ++j)
                stiffness(dofs_b[i], dofs_b[j]) += Kbb(i, j);
        }
    }
}

template class PenaltyCouplingCondition<double, 2>;

} // namespace pyck
