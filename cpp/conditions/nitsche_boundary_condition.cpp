#include "nitsche_boundary_condition.hpp"

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
NitscheBoundaryCondition<T, d>::NitscheBoundaryCondition(
    const PatchBoundary<T, d>& boundary,
    const Element<T, d>& element,
    const QuadratureRule<T, d - 1>& quadrature,
    T thickness)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature),
      thickness_(thickness)
{
    if (thickness_ <= T(0)) {
        throw std::invalid_argument("NitscheBoundaryCondition: thickness must be positive.");
    }
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
NitscheBoundaryCondition<T, d>& NitscheBoundaryCondition<T, d>::add(
    Ptr<const BoundaryField<T>> displacement_field,
    Ptr<const BoundaryField<T>> traction_field,
    T penalty,
    T value)
{
    if (!displacement_field) {
        throw std::invalid_argument(
            "NitscheBoundaryCondition::add: displacement_field must not be null.");
    }
    if (!traction_field) {
        throw std::invalid_argument(
            "NitscheBoundaryCondition::add: traction_field must not be null.");
    }
    terms_.push_back({std::move(displacement_field),
                      std::move(traction_field), penalty, value});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void NitscheBoundaryCondition<T, d>::apply(Matrix<T>& stiffness,
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
    // Traction fields can need higher derivatives than the element's stiffness:
    // KL's transverse-shear recovery uses 3rd derivatives, and bending-moment
    // projections need 2nd. Request whichever is larger.
    const std::size_t req_order = std::max<std::size_t>(element.min_order(), 3);

    const Index num_spans = boundary.basis(0).knot_vector().num_spans();
    const Index Q = static_cast<Index>(quadrature.num_points());
    ColMatrix<T, 1> mapped_pts(Q, 1);
    Vector<T>       mapped_weights(Q);
    for (Index span = 0; span < num_spans; ++span)
    {
        // Skip zero-length spans.
        auto [lo, hi] = boundary.basis(0).knot_vector().span_bounds(span);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        // Per-span scaffolding: shared across all terms.
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

        auto elem_dofs = parent.dof_mapper().get_element_dofs(flat_parent);
        const Index n_elem = static_cast<Index>(elem_dofs.size());
        const Index K_elem = n_elem * ndof;

        Matrix<T> K_local = Matrix<T>::Zero(K_elem, K_elem);
        Vector<T> F_local = Vector<T>::Zero(K_elem);

        // Accumulate every term's contribution into the same local matrices.
        for (const auto& term : terms_)
        {
            Matrix<T> N = term.displacement_field->evaluate(
                element, boundary, span, boundary_basis, boundary_local,
                flat_parent, parent_basis, parent_ig);
            Matrix<T> Qmat = term.traction_field->evaluate(
                element, boundary, span, boundary_basis, boundary_local,
                flat_parent, parent_basis, parent_ig);

            const T penalty_h = term.penalty / thickness_;

            for (Index q = 0; q < Q; ++q)
            {
                const T dGamma = boundary_local.jac(q) * mapped_weights(q);

                // Term 1: -N^T · Q (consistency)
                K_local.noalias() -= dGamma
                    * N.row(q).transpose() * Qmat.row(q);

                // Term 2: -Q^T · N (symmetry, stiffness part)
                K_local.noalias() -= dGamma
                    * Qmat.row(q).transpose() * N.row(q);

                // Term 2: Q^T · ū (symmetry, load part)
                F_local.noalias() -= dGamma * term.value
                    * Qmat.row(q).transpose();

                // Term 3: +(β/h)·N^T · N (penalty, stiffness part)
                K_local.noalias() += dGamma * penalty_h
                    * N.row(q).transpose() * N.row(q);

                // Term 3: +(β/h)·N^T · ū (penalty, load part)
                F_local.noalias() += dGamma * penalty_h * term.value
                    * N.row(q).transpose();
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

template class NitscheBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class NitscheBoundaryCondition<float, 2>;
#endif

} // namespace pyck
