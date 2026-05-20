#include "patch_boundary.hpp"
#include "patch.hpp"
#include "dof_mapper.hpp"
#include "intrinsic_geometry.hpp"

#include <array>
#include <cmath>

namespace pyck
{

namespace {

/// Construct the inherited Patch<T, d-1> base for a PatchBoundary<T, d>.
/// Gathers the d-1 free-direction basis pointers of the parent (skipping
/// `param_dim`) and unpacks them into the parent's variadic ctor.
template <std::floating_point T, std::size_t d>
Patch<T, d - 1>
make_boundary_patch(const Ptr<Patch<T, d>>& parent,
                    std::size_t param_dim,
                    bool at_start)
{
    auto cps = parent->get_control_points(
        parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0));

    std::array<Ptr<const Basis<T>>, d - 1> free_bases;
    std::size_t idx = 0;
    for (std::size_t k = 0; k < d; ++k) {
        if (k != param_dim) free_bases[idx++] = parent->basis_ptr(k);
    }

    if constexpr (d - 1 == 1) {
        return Patch<T, 1>(free_bases[0], cps);
    } else {
        return Patch<T, 2>(free_bases[0], free_bases[1], cps);
    }
}

/// sign convention for the outward boundary normal.
/// d=2: original convention `((param_dim == 1) == at_start) ? +1 : -1`.
/// d=3: derived so that `n = sign_n · (a_1^bd × a_2^bd) / ‖…‖` matches
///      the outward direction of an axis-aligned box face.
template <std::size_t d>
constexpr int boundary_sign(std::size_t param_dim, bool at_start)
{
    if constexpr (d == 2) {
        return ((param_dim == 1) == at_start) ? 1 : -1;
    } else {  // d == 3
        const int base = (param_dim % 2 == 0) ? 1 : -1;
        return (at_start ? -1 : 1) * base;
    }
}

}  // namespace

template <std::floating_point T, std::size_t d> requires (d > 1)
PatchBoundary<T, d>::PatchBoundary(const Ptr<Patch<T, d>>& parent,
                                   std::size_t param_dim,
                                   bool at_start)
    : Patch<T, d - 1>(make_boundary_patch<T, d>(parent, param_dim, at_start)),
      parent_(parent), param_dim_(param_dim), at_start_(at_start),
      u_eval_fixed_(T(0)),
      sign_n_(T(boundary_sign<d>(param_dim, at_start))),
      span_fixed_(0)
{
    if (param_dim >= d)
    {
        throw std::invalid_argument("PatchBoundary: "
                                    "param_dim is out of bounds for dimension d.");
    }

    parent_dofs_ = parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0);

    // Pick the first / last non-degenerate span along the fixed direction.
    auto& kv_fixed = parent->basis(param_dim).knot_vector();

    if (at_start)
    {
        for (Index s = 0; s < kv_fixed.num_spans(); ++s)
        {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) {
                span_fixed_ = s; u_eval_fixed_ = lo;
                break;
            }
        }
    }
    else
    {
        for (Index s = kv_fixed.num_spans(); s-- > 0;)
        {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) {
                span_fixed_ = s; u_eval_fixed_ = hi;
                break;
            }
        }
    }
}

template <std::floating_point T, std::size_t d> requires (d > 1)
Index PatchBoundary<T, d>::parent_flat_span(Index boundary_span) const
{
    // 1) Decode the boundary's flat span into per-direction boundary spans.
    auto boundary_spans = this->decode_span(boundary_span);

    // 2) Insert the fixed parent span at position `param_dim_` to form the
    //    parent span tuple.
    std::array<Index, d> parent_spans{};
    std::size_t bi = 0;
    for (std::size_t k = 0; k < d; ++k) {
        if (k == param_dim_) {
            parent_spans[k] = span_fixed_;
        } else {
            parent_spans[k] = boundary_spans[bi++];
        }
    }

    // 3) Lexicographic encode using the parent's interval counts (u-inner).
    const auto parent_intervals = parent_->tensor_product().num_intervals();
    Index flat = 0;
    Index stride = 1;
    for (std::size_t k = 0; k < d; ++k) {
        flat += parent_spans[k] * stride;
        stride *= parent_intervals[k];
    }
    return flat;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, d>
PatchBoundary<T, d>::lift_to_parent(const ColMatrix<T, d - 1>& boundary_pts) const
{
    const Index Q = boundary_pts.rows();
    ColMatrix<T, d> pts(Q, d);
    pts.col(param_dim_).setConstant(u_eval_fixed_);

    // Fill the d-1 free parametric directions (skipping param_dim_) from the
    // columns of boundary_pts in canonical order.
    Index bi = 0;
    for (std::size_t k = 0; k < d; ++k) {
        if (k == param_dim_) continue;
        pts.col(k) = boundary_pts.col(bi++);
    }
    return pts;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, 3>
PatchBoundary<T, d>::eval_outward_normal(const IntrinsicGeometry<T, d - 1>& boundary_local,
                                         const IntrinsicGeometry<T, d>& parent_local) const requires(d == 2)
{
    const Index Q = boundary_local.a(0).rows();
    ColMatrix<T, 3> n_mat(Q, 3);

    for (Index q = 0; q < Q; ++q)
    {
        // Parent surface normal a_3 = (a_1 × a_2) / ‖a_1 × a_2‖.
        Eigen::Matrix<T, 3, 1> pa1 = parent_local.a(0).row(q).transpose();
        Eigen::Matrix<T, 3, 1> pa2 = parent_local.a(1).row(q).transpose();
        Eigen::Matrix<T, 3, 1> a3 = pa1.cross(pa2);
        const T jac_p = parent_local.jac(q);
        if (jac_p > T(1e-14)) a3 /= jac_p;
        else a3 = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));

        // Outward in-surface normal n = sign_n · (a_1^bd × a_3) / ‖…‖.
        Eigen::Matrix<T, 3, 1> t = boundary_local.a(0).row(q).transpose();
        Eigen::Matrix<T, 3, 1> n_vec = sign_n_ * t.cross(a3);
        const T n_norm = n_vec.norm();
        if (n_norm > T(1e-14)) n_vec /= n_norm;
        n_mat.row(q) = n_vec.transpose();
    }
    return n_mat;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, 3>
PatchBoundary<T, d>::eval_outward_normal(const IntrinsicGeometry<T, d - 1>& boundary_local) const requires(d == 3)
{
    // Boundary is a 2D surface in 3D physical space; outward normal is the
    // boundary's own surface normal a_3 = a_1^bd × a_2^bd / ‖…‖, signed.
    const Index Q = boundary_local.a(0).rows();
    ColMatrix<T, 3> n_mat(Q, 3);

    for (Index q = 0; q < Q; ++q)
    {
        Eigen::Matrix<T, 3, 1> a1 = boundary_local.a(0).row(q).transpose();
        Eigen::Matrix<T, 3, 1> a2 = boundary_local.a(1).row(q).transpose();
        Eigen::Matrix<T, 3, 1> n_vec = sign_n_ * a1.cross(a2);
        const T n_norm = n_vec.norm();
        if (n_norm > T(1e-14)) n_vec /= n_norm;
        n_mat.row(q) = n_vec.transpose();
    }
    return n_mat;
}

// === Template Instantiations ========================================================

template class PatchBoundary<double, 2>;
template class PatchBoundary<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PatchBoundary<float, 2>;
template class PatchBoundary<float, 3>;
#endif

} // namespace pyck
