#include "integrate.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "../geometry/patch.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
Vector<T> integrate_on_patch(
    const Patch<T, d>& patch,
    const QuadratureRule<T, d>& quadrature,
    std::size_t order,
    const Vector<T>& cp_values,
    std::size_t ndof,
    const std::function<Matrix<T>(
        const ColMatrix<T, 3>&,
        const std::vector<Matrix<T>>&,
        const Vector<T>&)>& integrand)
{
    const bool has_cps = (cp_values.size() > 0);
    if (has_cps) {
        if (ndof == 0) {
            throw std::runtime_error("integrate_on_patch: ndof must be > 0 when cp_values is provided.");
        }
        const Index n_cp = static_cast<Index>(patch.num_control_pts());
        const Index expected = n_cp * static_cast<Index>(ndof);
        if (cp_values.size() != expected) {
            throw std::runtime_error(
                "integrate_on_patch: cp_values has size " + std::to_string(cp_values.size()) +
                " but expected n_cp * ndof = " + std::to_string(expected) + ".");
        }
    }

    std::array<std::size_t, d> intervals;
    std::size_t total_elements = 1;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = patch.basis(i).knot_vector().num_spans();
        total_elements *= intervals[i];
    }

    const std::size_t Q = quadrature.num_points();
    const Vector<T> empty_u;  // size 0, passed when has_cps is false

    Vector<T> result;          // sized on first integrand call
    bool result_initialised = false;

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

        auto [shape_derivs, jacobian] = patch.eval_shape_functions(
            mapped_pts, static_cast<Index>(elem_idx), order);

        ColMatrix<T, 3> phys_pts = patch.eval_geometry(
            mapped_pts, static_cast<Index>(elem_idx));

        Vector<T> u_local;
        if (has_cps) {
            auto elem_nodes = patch.dof_mapper().get_element_dofs(
                static_cast<Index>(elem_idx));
            u_local.resize(static_cast<Index>(elem_nodes.size() * ndof));
            for (std::size_t k = 0; k < elem_nodes.size(); ++k) {
                for (std::size_t v = 0; v < ndof; ++v) {
                    u_local(static_cast<Index>(k * ndof + v)) =
                        cp_values(static_cast<Index>(elem_nodes[k] * ndof + v));
                }
            }
        }

        Matrix<T> values = integrand(phys_pts, shape_derivs,
                                     has_cps ? u_local : empty_u);

        if (values.rows() != static_cast<Index>(Q)) {
            throw std::runtime_error(
                "integrate_on_patch: integrand returned " + std::to_string(values.rows()) +
                " rows but the quadrature has " + std::to_string(Q) + " points.");
        }

        if (!result_initialised) {
            result = Vector<T>::Zero(values.cols());
            result_initialised = true;
        } else if (result.size() != values.cols()) {
            throw std::runtime_error(
                "integrate_on_patch: integrand returned " + std::to_string(values.cols()) +
                " components on this element but " + std::to_string(result.size()) +
                " on a previous one.");
        }

        for (std::size_t k = 0; k < Q; ++k) {
            const T dV = mapped_weights(k) * jacobian(k);
            result.noalias() += dV * values.row(static_cast<Index>(k)).transpose();
        }
    }

    if (!result_initialised) {
        return Vector<T>(0);
    }
    return result;
}

// === Template Instantiations ========================================================

template Vector<double> integrate_on_patch<double, 1>(
    const Patch<double, 1>&,
    const QuadratureRule<double, 1>&,
    std::size_t,
    const Vector<double>&,
    std::size_t,
    const std::function<Matrix<double>(
        const ColMatrix<double, 3>&,
        const std::vector<Matrix<double>>&,
        const Vector<double>&)>&);

template Vector<double> integrate_on_patch<double, 2>(
    const Patch<double, 2>&,
    const QuadratureRule<double, 2>&,
    std::size_t,
    const Vector<double>&,
    std::size_t,
    const std::function<Matrix<double>(
        const ColMatrix<double, 3>&,
        const std::vector<Matrix<double>>&,
        const Vector<double>&)>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template Vector<float> integrate_on_patch<float, 1>(
    const Patch<float, 1>&,
    const QuadratureRule<float, 1>&,
    std::size_t,
    const Vector<float>&,
    std::size_t,
    const std::function<Matrix<float>(
        const ColMatrix<float, 3>&,
        const std::vector<Matrix<float>>&,
        const Vector<float>&)>&);

template Vector<float> integrate_on_patch<float, 2>(
    const Patch<float, 2>&,
    const QuadratureRule<float, 2>&,
    std::size_t,
    const Vector<float>&,
    std::size_t,
    const std::function<Matrix<float>(
        const ColMatrix<float, 3>&,
        const std::vector<Matrix<float>>&,
        const Vector<float>&)>&);
#endif

} // namespace pyck
