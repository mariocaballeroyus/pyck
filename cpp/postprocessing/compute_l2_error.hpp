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
#include "../geometry/basis_derivs.hpp"
#include "../geometry/local_frame.hpp"
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

        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(u_a, u_b);

        auto basis   = eval_basis(patch, mapped_pts, static_cast<Index>(elem_idx), basis_order);
        auto act_pts = patch.active_control_pts(static_cast<Index>(elem_idx));
        auto local   = eval_local_frame(basis, act_pts);

        const ColMatrix<T, 3> phys_pts = basis.N * act_pts;

        const auto elem_nodes = patch.dof_mapper().get_element_dofs(
            static_cast<Index>(elem_idx));
        Vector<T> u_local(static_cast<Index>(elem_nodes.size() * ndof));
        for (std::size_t k = 0; k < elem_nodes.size(); ++k) {
            for (std::size_t v = 0; v < ndof; ++v) {
                u_local(static_cast<Index>(k * ndof + v)) =
                    num_u(static_cast<Index>(elem_nodes[k] * ndof + v));
            }
        }

        std::vector<Matrix<T>> shape_derivs;
        if constexpr (d == 1) {
            shape_derivs.push_back(basis.N);
            if (order >= 1) shape_derivs.push_back(basis.N_u);
            if (order >= 2) shape_derivs.push_back(basis.N_uu);
            if (order >= 3) shape_derivs.push_back(basis.N_uuu);
        } else {
            shape_derivs.push_back(basis.N);
            if (order >= 1) {
                shape_derivs.push_back(basis.N_u);
                shape_derivs.push_back(basis.N_v);
            }
            if (order >= 2) {
                shape_derivs.push_back(basis.N_uu);
                shape_derivs.push_back(basis.N_uv);
                shape_derivs.push_back(basis.N_vv);
            }
            if (order >= 3) {
                shape_derivs.push_back(basis.N_uuu);
                shape_derivs.push_back(basis.N_uuv);
                shape_derivs.push_back(basis.N_uvv);
                shape_derivs.push_back(basis.N_vvv);
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
