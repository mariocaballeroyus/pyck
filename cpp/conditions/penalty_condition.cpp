#include "penalty_condition.hpp"
#include "boundary_patch.hpp"
#include "patch.hpp"
#include "bspline.hpp"

#include <cmath>
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
PenaltyCondition<T>::PenaltyCondition(
    const BoundaryPatch<T, 2>& boundary,
    const Element<T, 2>&       element,
    const QuadratureRule<T, 1>& quadrature,
    T alpha_w,    T w_bar,
    T alpha_phi_n, T phi_n_bar,
    T alpha_phi_s, T phi_s_bar)
{
    const auto& parent  = *boundary.parent();
    const std::size_t p_dim    = boundary.param_dim();
    const bool        at_start = boundary.at_start();
    const Index       ndof     = element.num_node_dofs();
    const std::size_t req_order = element.min_order();

    // ------------------------------------------------------------------
    // Fixed-direction span (the constant-u or constant-v side of Γ).
    // Used as the second index into the 2D flat span of the parent.
    // ------------------------------------------------------------------
    auto& kv_fixed = parent.basis(p_dim).knot_vector();
    Index span_fixed   = 0;
    T     u_eval_fixed = T(0);

    if (at_start) {
        for (Index s = 0; s < kv_fixed.num_spans(); ++s) {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) {
                span_fixed   = s;
                u_eval_fixed = lo;
                break;
            }
        }
    } else {
        for (Index s = kv_fixed.num_spans(); s-- > 0; ) {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) {
                span_fixed   = s;
                u_eval_fixed = hi;
                break;
            }
        }
    }

    // Outward-normal sign, derived from n = sign_n * (t × a3).
    const T sign_n = ((p_dim == 1) == at_start) ? T(1) : T(-1);

    const Index n_int_u = parent.tensor_product().num_intervals()[0];

    // ------------------------------------------------------------------
    // Loop over 1D boundary elements (knot spans of the BoundaryPatch)
    // ------------------------------------------------------------------
    const Index num_spans_bdy = boundary.basis(0).knot_vector().num_spans();

    for (Index s = 0; s < num_spans_bdy; ++s)
    {
        auto [lo, hi] = boundary.basis(0).knot_vector().span_bounds(s);
        if (std::abs(hi - lo) < T(1e-14)) continue;

        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        const Index Q = static_cast<Index>(mapped_pts.rows());

        // 1D boundary shape functions (only used for the unit tangent).
        auto [sfd_bdy, jac_bdy] =
            boundary.eval_shape_functions(mapped_pts, s, 1);

        auto spans_bdy   = boundary.decode_span(s);
        auto act_pts_bdy = boundary.active_control_pts(spans_bdy);

        // (Q × 3) unit tangent in the boundary parametric direction
        ColMatrix<T, 3> t_mat = sfd_bdy[1] * act_pts_bdy;

        // ------------------------------------------------------------------
        // Locate the parent 2D element (flat span) covering this slice, and
        // evaluate the parent basis with `req_order` derivatives at every
        // boundary quadrature point.  All Q points lie in the same 2D span.
        // ------------------------------------------------------------------
        const T v_eval_mid = (lo + hi) / T(2);
        const Index span_free = parent.basis(1 - p_dim).find_span(v_eval_mid);
        const Index flat_parent = (p_dim == 0)
            ? (span_fixed + span_free * n_int_u)
            : (span_free  + span_fixed * n_int_u);

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

        auto [sfd_par, jac_par] =
            parent.eval_shape_functions(pts_2d, flat_parent, req_order);
        auto elem_dofs = parent.dof_mapper().get_element_dofs(flat_parent);
        const Index n_elem = static_cast<Index>(elem_dofs.size());
        const Index K_elem = n_elem * ndof;

        // ------------------------------------------------------------------
        // Surface normal a_3 from parent geometry at the span midpoint.
        // For flat plates a_3 = (0,0,1) and the computation is exact; for
        // curved surfaces it is evaluated once per span (midpoint).
        // ------------------------------------------------------------------
        ColMatrix<T, 2> pt_mid(1, 2);
        if (p_dim == 0) { pt_mid(0, 0) = u_eval_fixed; pt_mid(0, 1) = v_eval_mid; }
        else            { pt_mid(0, 0) = v_eval_mid;  pt_mid(0, 1) = u_eval_fixed; }
        auto [sfd_mid, jac_mid] = parent.eval_shape_functions(pt_mid, flat_parent, 1);
        auto spans_par_mid   = parent.decode_span(flat_parent);
        auto act_pts_par_mid = parent.active_control_pts(spans_par_mid);

        Eigen::Matrix<T, 3, 1> a1_u = (sfd_mid[1] * act_pts_par_mid).row(0).transpose();
        Eigen::Matrix<T, 3, 1> a2_u = (sfd_mid[2] * act_pts_par_mid).row(0).transpose();
        Eigen::Matrix<T, 3, 1> a3   = a1_u.cross(a2_u);
        const T a3_norm = a3.norm();
        if (a3_norm > T(1e-14)) a3 /= a3_norm;
        else                     a3 = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));

        // ------------------------------------------------------------------
        // Full-DOF-layout shape matrices: Nw (Q × K_elem), Nphi (2Q × K_elem).
        // ------------------------------------------------------------------
        Matrix<T> Nw   = element.displacement_shape_matrix(sfd_par);
        Matrix<T> Nphi = element.rotation_shape_matrix(sfd_par);

        Matrix<T> K_local = Matrix<T>::Zero(K_elem, K_elem);
        Vector<T> F_local = Vector<T>::Zero(K_elem);

        for (Index q = 0; q < Q; ++q)
        {
            const T dGamma = jac_bdy(q) * mapped_weights(q);

            Eigen::Matrix<T, 3, 1> t_q = t_mat.row(q).transpose();
            Eigen::Matrix<T, 3, 1> n_vec = sign_n * t_q.cross(a3);
            const T n_x = n_vec(0);
            const T n_y = n_vec(1);
            // In-plane tangent s ⊥ n, with (s, n, a3) right-handed.
            const T s_x = -n_y;
            const T s_y =  n_x;

            auto Nw_row  = Nw.row(q);
            auto Ntx_row = Nphi.row(2*q    );
            auto Nty_row = Nphi.row(2*q + 1);

            // θ_n = n·θ, θ_s = s·θ  (1 × K_elem each).
            Eigen::Matrix<T, 1, Eigen::Dynamic> Nn_row = n_x * Ntx_row + n_y * Nty_row;
            Eigen::Matrix<T, 1, Eigen::Dynamic> Ns_row = s_x * Ntx_row + s_y * Nty_row;

            K_local.noalias() += dGamma *
                (alpha_w    * Nw_row.transpose() * Nw_row
               + alpha_phi_n * Nn_row.transpose() * Nn_row
               + alpha_phi_s * Ns_row.transpose() * Ns_row);

            F_local.noalias() += dGamma *
                (alpha_w    * w_bar      * Nw_row.transpose()
               + alpha_phi_n * phi_n_bar * Nn_row.transpose()
               + alpha_phi_s * phi_s_bar * Ns_row.transpose());
        }

        // Expand element DOFs into the full DOF layout and stash.
        std::vector<Index> local_global;
        local_global.reserve(K_elem);
        for (auto cp : elem_dofs) {
            for (Index v = 0; v < ndof; ++v) {
                local_global.push_back(cp * ndof + v);
            }
        }

        contributions_.push_back({std::move(K_local),
                                  std::move(F_local),
                                  std::move(local_global)});
    }
}

template <std::floating_point T>
void PenaltyCondition<T>::apply(Matrix<T>& stiffness, Vector<T>& load) const
{
    for (const auto& c : contributions_) {
        const Index n = static_cast<Index>(c.dofs.size());
        for (Index i = 0; i < n; ++i) {
            const Index gi = c.dofs[i];
            load(gi) += c.F(i);
            for (Index j = 0; j < n; ++j) {
                stiffness(gi, c.dofs[j]) += c.K(i, j);
            }
        }
    }
}

// === Template Instantiations =====================================================

template class PenaltyCondition<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PenaltyCondition<float>;
#endif

} // namespace pyck
