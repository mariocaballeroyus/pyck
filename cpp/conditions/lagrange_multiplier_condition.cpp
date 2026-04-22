#include "lagrange_multiplier_condition.hpp"

#include "boundary_patch.hpp"
#include "bspline.hpp"
#include "patch.hpp"

#include <cmath>
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
LagrangeMultiplierCondition<T>::LagrangeMultiplierCondition(
    const BoundaryPatch<T, 2>& boundary,
    const Element<T, 2>& element,
    const QuadratureRule<T, 1>& quadrature,
    bool enforce_w, T w_bar,
    bool enforce_phi_n, T phi_n_bar,
    bool enforce_phi_s, T phi_s_bar)
{
    if (!enforce_w && !enforce_phi_n && !enforce_phi_s) {
        throw std::invalid_argument(
            "LagrangeMultiplierCondition: at least one boundary field must be enforced."
        );
    }

    boundary_basis_count_ = static_cast<Index>(boundary.num_control_pts());
    if (boundary_basis_count_ == 0) {
        throw std::invalid_argument(
            "LagrangeMultiplierCondition: boundary has no active basis functions."
        );
    }

    const auto rotation_dofs = element.rotation_dof_indices();
    const bool derived_rotation_plate = (rotation_dofs[0] == rotation_dofs[1]);
    const bool need_derivative_basis = derived_rotation_plate && enforce_phi_s;
    if (need_derivative_basis) {
        auto boundary_bspline =
            std::dynamic_pointer_cast<const BSpline<T>>(boundary.basis_ptr(0));
        if (!boundary_bspline) {
            throw std::invalid_argument(
                "LagrangeMultiplierCondition: tangential-rotation multipliers for "
                "plates with derived rotations require a BSpline boundary basis."
            );
        }

        if (boundary_bspline->degree() < 1) {
            throw std::invalid_argument(
                "LagrangeMultiplierCondition: tangential-rotation multiplier basis "
                "requires boundary degree >= 1."
            );
        }

        const auto& knots = boundary_bspline->knot_vector().data();
        std::vector<T> deriv_knots(knots.begin() + 1, knots.end() - 1);
        derivative_basis_ = std::make_shared<BSpline<T>>(
            boundary_bspline->degree() - 1,
            KnotVector<T>(std::move(deriv_knots))
        );
        derivative_basis_count_ = derivative_basis_->num_basis();
        derivative_dof_mapper_.emplace(
            std::array<Index, 1>{derivative_basis_count_},
            std::array<Index, 1>{static_cast<Index>(derivative_basis_->degree())}
        );
    }

    if (enforce_w) {
        components_.push_back({
            ComponentKind::displacement,
            w_bar,
            0,
            boundary_basis_count_,
            false,
        });
    }
    if (enforce_phi_n) {
        components_.push_back({
            ComponentKind::normal_rotation,
            phi_n_bar,
            0,
            boundary_basis_count_,
            false,
        });
    }
    if (enforce_phi_s) {
        components_.push_back({
            ComponentKind::tangential_rotation,
            phi_s_bar,
            0,
            derived_rotation_plate ? derivative_basis_count_ : boundary_basis_count_,
            derived_rotation_plate,
        });
    }

    Index next_offset = 0;
    for (auto& component : components_) {
        component.block_offset = next_offset;
        next_offset += component.dof_count;
    }
    num_multipliers_ = next_offset;

    const auto& parent = *boundary.parent();
    const std::size_t p_dim = boundary.param_dim();
    const bool at_start = boundary.at_start();
    const Index ndof = static_cast<Index>(element.num_node_dofs());
    const std::size_t req_order = element.min_order();

    auto& kv_fixed = parent.basis(p_dim).knot_vector();
    Index span_fixed = 0;
    T u_eval_fixed = T(0);

    if (at_start) {
        for (Index s = 0; s < kv_fixed.num_spans(); ++s) {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) {
                span_fixed = s;
                u_eval_fixed = lo;
                break;
            }
        }
    } else {
        for (Index s = kv_fixed.num_spans(); s-- > 0;) {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) {
                span_fixed = s;
                u_eval_fixed = hi;
                break;
            }
        }
    }

    const T sign_n = ((p_dim == 1) == at_start) ? T(1) : T(-1);
    const Index n_int_u = parent.tensor_product().num_intervals()[0];
    const Index num_spans_bdy = boundary.basis(0).knot_vector().num_spans();

    for (Index s = 0; s < num_spans_bdy; ++s)
    {
        auto [lo, hi] = boundary.basis(0).knot_vector().span_bounds(s);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        const Index Q = static_cast<Index>(mapped_pts.rows());

        auto [sfd_bdy, jac_bdy] = boundary.eval_shape_functions(mapped_pts, s, 1);
        auto trace_basis_ids = boundary.dof_mapper().get_element_dofs(s);
        const Matrix<T>& M_trace = sfd_bdy[0];

        std::vector<Index> derivative_basis_ids;
        Matrix<T> M_derivative;
        if (derivative_basis_) {
            const Index deriv_span = derivative_basis_->find_span((lo + hi) / T(2));
            derivative_basis_ids = derivative_dof_mapper_->get_element_dofs(deriv_span);
            M_derivative = derivative_basis_->eval_on_span(mapped_pts, deriv_span, 0)[0];
        }

        auto spans_bdy = boundary.decode_span(s);
        auto act_pts_bdy = boundary.active_control_pts(spans_bdy);
        ColMatrix<T, 3> t_mat = sfd_bdy[1] * act_pts_bdy;

        const T v_eval_mid = (lo + hi) / T(2);
        const Index span_free = parent.basis(1 - p_dim).find_span(v_eval_mid);
        const Index flat_parent = (p_dim == 0)
            ? (span_fixed + span_free * n_int_u)
            : (span_free + span_fixed * n_int_u);

        ColMatrix<T, 2> pts_2d(Q, 2);
        for (Index q = 0; q < Q; ++q) {
            if (p_dim == 0) {
                pts_2d(q, 0) = u_eval_fixed;
                pts_2d(q, 1) = mapped_pts(q);
            } else {
                pts_2d(q, 0) = mapped_pts(q);
                pts_2d(q, 1) = u_eval_fixed;
            }
        }

        auto [sfd_par, jac_par] = parent.eval_shape_functions(pts_2d, flat_parent, req_order);
        auto elem_dofs = parent.dof_mapper().get_element_dofs(flat_parent);
        const Index n_elem = static_cast<Index>(elem_dofs.size());
        const Index K_elem = n_elem * ndof;

        ColMatrix<T, 2> pt_mid(1, 2);
        if (p_dim == 0) {
            pt_mid(0, 0) = u_eval_fixed;
            pt_mid(0, 1) = v_eval_mid;
        } else {
            pt_mid(0, 0) = v_eval_mid;
            pt_mid(0, 1) = u_eval_fixed;
        }

        auto [sfd_mid, jac_mid] = parent.eval_shape_functions(pt_mid, flat_parent, 1);
        auto spans_par_mid = parent.decode_span(flat_parent);
        auto act_pts_par_mid = parent.active_control_pts(spans_par_mid);

        Eigen::Matrix<T, 3, 1> a1_u = (sfd_mid[1] * act_pts_par_mid).row(0).transpose();
        Eigen::Matrix<T, 3, 1> a2_u = (sfd_mid[2] * act_pts_par_mid).row(0).transpose();
        Eigen::Matrix<T, 3, 1> a3 = a1_u.cross(a2_u);
        const T a3_norm = a3.norm();
        if (a3_norm > T(1e-14)) a3 /= a3_norm;
        else a3 = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));

        Matrix<T> Nw = element.displacement_shape_matrix(sfd_par);
        Matrix<T> Nphi = element.rotation_shape_matrix(sfd_par);

        std::vector<Index> local_row_offsets(components_.size(), 0);
        Index local_row_count = 0;
        for (Index c = 0; c < static_cast<Index>(components_.size()); ++c) {
            local_row_offsets[c] = local_row_count;
            local_row_count += components_[c].use_derivative_basis
                ? static_cast<Index>(derivative_basis_ids.size())
                : static_cast<Index>(trace_basis_ids.size());
        }

        Matrix<T> C_local = Matrix<T>::Zero(local_row_count, K_elem);
        Vector<T> G_local = Vector<T>::Zero(local_row_count);

        for (Index q = 0; q < Q; ++q)
        {
            const T dGamma = jac_bdy(q) * mapped_weights(q);

            Eigen::Matrix<T, 3, 1> t_q = t_mat.row(q).transpose();
            Eigen::Matrix<T, 3, 1> n_vec = sign_n * t_q.cross(a3);
            const T n_x = n_vec(0);
            const T n_y = n_vec(1);
            const T s_x = -n_y;
            const T s_y = n_x;

            auto Nw_row = Nw.row(q);
            auto Ntx_row = Nphi.row(2 * q);
            auto Nty_row = Nphi.row(2 * q + 1);

            Eigen::Matrix<T, 1, Eigen::Dynamic> Nn_row = n_x * Ntx_row + n_y * Nty_row;
            Eigen::Matrix<T, 1, Eigen::Dynamic> Ns_row = s_x * Ntx_row + s_y * Nty_row;

            for (Index c = 0; c < static_cast<Index>(components_.size()); ++c) {
                const auto& component = components_[c];
                Eigen::Matrix<T, 1, Eigen::Dynamic> constraint_row(1, K_elem);
                switch (component.kind) {
                case ComponentKind::displacement:
                    constraint_row = Nw_row;
                    break;
                case ComponentKind::normal_rotation:
                    constraint_row = Nn_row;
                    break;
                case ComponentKind::tangential_rotation:
                    constraint_row = Ns_row;
                    break;
                }

                const auto& basis_ids = component.use_derivative_basis
                    ? derivative_basis_ids
                    : trace_basis_ids;
                const auto& M = component.use_derivative_basis
                    ? M_derivative
                    : M_trace;
                const Index n_lambda = static_cast<Index>(basis_ids.size());
                const Index row_offset = local_row_offsets[c];

                C_local.middleRows(row_offset, n_lambda) +=
                    dGamma * M.row(q).transpose() * constraint_row;
                G_local.segment(row_offset, n_lambda) +=
                    dGamma * component.prescribed_value * M.row(q).transpose();
            }
        }

        std::vector<Index> primal_global;
        primal_global.reserve(K_elem);
        for (auto cp : elem_dofs) {
            for (Index v = 0; v < ndof; ++v) {
                primal_global.push_back(cp * ndof + v);
            }
        }

        std::vector<Index> lambda_local;
        lambda_local.reserve(static_cast<std::size_t>(local_row_count));
        for (const auto& component : components_) {
            const auto& basis_ids = component.use_derivative_basis
                ? derivative_basis_ids
                : trace_basis_ids;
            for (auto basis_id : basis_ids) {
                lambda_local.push_back(component.block_offset + basis_id);
            }
        }

        contributions_.push_back({
            std::move(C_local),
            std::move(G_local),
            std::move(primal_global),
            std::move(lambda_local),
        });
    }
}

template <std::floating_point T>
void LagrangeMultiplierCondition<T>::apply(Matrix<T>& stiffness, Vector<T>& load) const
{
    for (const auto& c : contributions_) {
        const Index n_lambda = static_cast<Index>(c.lambda_dofs.size());
        const Index n_primal = static_cast<Index>(c.primal_dofs.size());

        for (Index i = 0; i < n_lambda; ++i) {
            const Index lambda_i = static_cast<Index>(this->multiplier_offset() + c.lambda_dofs[i]);
            load(lambda_i) += c.G(i);

            for (Index j = 0; j < n_primal; ++j) {
                const Index primal_j = c.primal_dofs[j];
                const T cij = c.C(i, j);
                stiffness(lambda_i, primal_j) += cij;
                stiffness(primal_j, lambda_i) += cij;
            }
        }
    }
}

template class LagrangeMultiplierCondition<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LagrangeMultiplierCondition<float>;
#endif

} // namespace pyck
