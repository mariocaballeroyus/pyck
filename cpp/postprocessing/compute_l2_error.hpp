#ifndef PYCK_COMPUTE_L2_ERROR_HPP
#define PYCK_COMPUTE_L2_ERROR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../elements/element.hpp"
#include "../basis/tensor_product.hpp"
#include "../geometry/intrinsic_geometry.hpp"
#include "../geometry/patch.hpp"
#include "../quadrature/quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Component-wise L2 norms of a user-defined squared-error integrand.
 *
 * For each column c of the integrand's per-qpt output the function returns
 *   sqrt( ∫_Ω integrand(x, basis, u_local)[:, c] dΩ ).
 *
 * Per element: builds the active basis derivatives, gathers the active DOF
 * block from `num_u`, computes the physical coords, calls the integrand,
 * and accumulates each column with dV = w · |J|.
 *
 * @param patch       Patch over which the integral is taken.
 * @param element     Element formulation (provides ndof per node).
 * @param quadrature  Reference quadrature rule, mapped per element.
 * @param order       Highest derivative order to expose to the integrand.
 * @param num_u       Numerical DOF vector, length n_cp · element.num_node_dofs().
 * @param integrand   Callable(phys_pts (Q,3), shape_derivs, u_local) -> (Q, n)
 *                    matrix of per-qpt squared values. `shape_derivs` is a
 *                    list of raw parametric derivative matrices in canonical
 *                    order (1D: [N, N_u, N_uu, N_uuu]; 2D: [N, N_u, N_v,
 *                    N_uu, N_uv, N_vv, N_uuu, N_uuv, N_uvv, N_vvv]),
 *                    truncated to `order`.
 * @return Vector of length n: per-column L2 norms.
 */
template <std::floating_point T, std::size_t d>
Vector<T> compute_l2_error(
    const Patch<T, d>& patch,
    const Element<T, d>& element,
    const QuadratureRule<T, d>& quadrature,
    std::size_t order,
    const Vector<T>& num_u,
    const std::function<Matrix<T>(
        const ColMatrix<T, 3>&,
        const std::vector<Matrix<T>>&,
        const Vector<T>&)>& integrand)
{
    const std::size_t ndof = element.num_node_dofs();
    const std::size_t basis_order = std::max<std::size_t>(order, 1);

    std::array<std::size_t, d> intervals;
    std::size_t total_elements = 1;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = patch.basis(i).knot_vector().num_spans();
        total_elements *= intervals[i];
    }

    Vector<T> accum;
    bool initialised = false;

    const Index Q = static_cast<Index>(quadrature.num_points());
    ColMatrix<T, d> mapped_pts(Q, static_cast<Index>(d));
    Vector<T>       mapped_weights(Q);

    for (std::size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx)
    {
        std::array<std::size_t, d> span_indices;
        std::size_t temp = elem_idx;
        for (std::size_t i = 0; i < d; ++i) {
            span_indices[i] = temp % intervals[i];
            temp /= intervals[i];
        }

        std::array<T, d> u_a, u_b;
        bool zero_volume = false;
        for (std::size_t i = 0; i < d; ++i) {
            auto [lo, hi] = patch.basis(i).knot_vector().span_bounds(span_indices[i]);
            u_a[i] = lo;
            u_b[i] = hi;
            if (std::abs(hi - lo) < T(1e-14)) {
                zero_volume = true;
                break;
            }
        }
        if (zero_volume) continue;

        quadrature.map_to_domain(u_a, u_b, mapped_pts, mapped_weights);

        auto basis   = patch.tensor_product().eval_all(mapped_pts, basis_order);
        auto act_pts = patch.active_control_pts(static_cast<Index>(elem_idx));
        IntrinsicGeometry<T, d> local(basis, act_pts);

        const Index Q_eval = basis[0].cols();
        const Index N_eval = basis[0].rows();

        // Physical points: x(q) = Σ_i N_i(u_q) · P_i.
        ColMatrix<T, 3> phys_pts(Q_eval, 3);
        for (Index q = 0; q < Q_eval; ++q) {
            auto slab0 = basis[0].col(q);
            Eigen::Matrix<T, 1, 3> x_q = Eigen::Matrix<T, 1, 3>::Zero();
            for (Index i = 0; i < N_eval; ++i) {
                x_q.noalias() += slab0(i) * act_pts.row(i);
            }
            phys_pts.row(q) = x_q;
        }

        std::vector<Index> elem_nodes;
        patch.dof_mapper().get_element_dofs(
            static_cast<Index>(elem_idx), elem_nodes);
        Vector<T> u_local(static_cast<Index>(elem_nodes.size() * ndof));
        for (std::size_t k = 0; k < elem_nodes.size(); ++k) {
            for (std::size_t v = 0; v < ndof; ++v) {
                u_local(static_cast<Index>(k * ndof + v)) =
                    num_u(static_cast<Index>(elem_nodes[k] * ndof + v));
            }
        }

        // Build a concrete Q × N matrix M(q, i) = ∂_{packed in order k} B_i(u_q),
        // sourced from the slab data of `basis`. n_k is the per-basis stride
        // within order k (= num_multi_indices<d>(k)).
        auto extract_deriv = [&](Index k_order, Index packed_in_order, Index n_k) {
            Matrix<T> M(Q_eval, N_eval);
            for (Index q = 0; q < Q_eval; ++q) {
                auto slab = basis[k_order].col(q);
                for (Index i = 0; i < N_eval; ++i) {
                    M(q, i) = slab(i * n_k + packed_in_order);
                }
            }
            return M;
        };

        std::vector<Matrix<T>> shape_derivs;
        if constexpr (d == 1) {
            shape_derivs.push_back(extract_deriv(0, 0, 1));   // N
            if (order >= 1) shape_derivs.push_back(extract_deriv(1, 0, 1));   // ∂x N
            if (order >= 2) shape_derivs.push_back(extract_deriv(2, 0, 1));   // ∂xx N
            if (order >= 3) shape_derivs.push_back(extract_deriv(3, 0, 1));   // ∂xxx N
        } else {
            constexpr Index n_d1 = 2;
            constexpr Index n_d2 = 3;
            constexpr Index n_d3 = 4;
            shape_derivs.push_back(extract_deriv(0, 0, 1));        // N
            if (order >= 1) {
                shape_derivs.push_back(extract_deriv(1, 0, n_d1)); // ∂u N
                shape_derivs.push_back(extract_deriv(1, 1, n_d1)); // ∂v N
            }
            if (order >= 2) {
                shape_derivs.push_back(extract_deriv(2, 0, n_d2)); // ∂uu N
                shape_derivs.push_back(extract_deriv(2, 2, n_d2)); // ∂uv N (Voigt: (0,1) → 2)
                shape_derivs.push_back(extract_deriv(2, 1, n_d2)); // ∂vv N (Voigt: (1,1) → 1)
            }
            if (order >= 3) {
                shape_derivs.push_back(extract_deriv(3, 0, n_d3)); // ∂uuu
                shape_derivs.push_back(extract_deriv(3, 1, n_d3)); // ∂uuv
                shape_derivs.push_back(extract_deriv(3, 2, n_d3)); // ∂uvv
                shape_derivs.push_back(extract_deriv(3, 3, n_d3)); // ∂vvv
            }
        }

        const Matrix<T> values = integrand(phys_pts, shape_derivs, u_local);

        const Index Q = static_cast<Index>(mapped_weights.size());
        if (values.rows() != Q) {
            throw std::runtime_error(
                "compute_l2_error: integrand returned " + std::to_string(values.rows()) +
                " rows but the quadrature has " + std::to_string(Q) + " points.");
        }

        if (!initialised) {
            accum = Vector<T>::Zero(values.cols());
            initialised = true;
        } else if (accum.size() != values.cols()) {
            throw std::runtime_error(
                "compute_l2_error: integrand returned " + std::to_string(values.cols()) +
                " components on this element but " + std::to_string(accum.size()) +
                " on a previous one.");
        }

        for (Index q = 0; q < Q; ++q) {
            const T dV = mapped_weights(q) * local.jac(q);
            accum.noalias() += dV * values.row(q).transpose();
        }
    }

    if (!initialised) return Vector<T>(0);

    Vector<T> result(accum.size());
    for (Index i = 0; i < accum.size(); ++i) {
        result(i) = std::sqrt(accum(i));
    }
    return result;
}

} // namespace pyck

#endif // PYCK_COMPUTE_L2_ERROR_HPP
