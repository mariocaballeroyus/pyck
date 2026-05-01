#include "boundary_patch.hpp"
#include "patch.hpp"
#include "factories.hpp"
#include "factories.hpp"
#include "dof_mapper.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d> requires (d > 1)
BoundaryPatch<T, d>::BoundaryPatch(const Ptr<Patch<T, d>>& parent,
                                   std::size_t param_dim,
                                   bool at_start)
    : Patch<T, d - 1>(parent->basis_ptr(1 - param_dim),
                    parent->get_control_points(parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0))),
      parent_(parent), param_dim_(param_dim), at_start_(at_start),
      span_fixed_(0), u_eval_fixed_(T(0)),
      sign_n_(((param_dim == 1) == at_start) ? T(1) : T(-1))
{
    if (param_dim >= d) {
        throw std::invalid_argument("BoundaryPatch: "
                                    "param_dim is out of bounds for dimension d."
        );
    }

    parent_dofs_ = parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0);

    auto& kv_fixed = parent->basis(param_dim).knot_vector();
    if (at_start) {
        for (Index s = 0; s < kv_fixed.num_spans(); ++s) {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) { span_fixed_ = s; u_eval_fixed_ = lo; break; }
        }
    } else {
        for (Index s = kv_fixed.num_spans(); s-- > 0;) {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) { span_fixed_ = s; u_eval_fixed_ = hi; break; }
        }
    }

    const Index n_int_u = parent->tensor_product().num_intervals()[0];
    parent_span_offset_ = (param_dim_ == 0) ? span_fixed_ : span_fixed_ * n_int_u;
    parent_span_stride_ = (param_dim_ == 0) ? n_int_u : Index(1);
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, 3> BoundaryPatch<T, d>::eval_normal(
    const std::array<ColMatrix<T, 3>, d - 1>& tangents,
    T sign_n,
    const std::array<ColMatrix<T, 3>, d>& parent_tangents) const requires(d == 2)
{
    const Index Q = tangents[0].rows();
    ColMatrix<T, 3> n_mat(Q, 3);
    for (Index q = 0; q < Q; ++q) {
        Eigen::Matrix<T, 3, 1> a1_q = parent_tangents[0].row(q).transpose();
        Eigen::Matrix<T, 3, 1> a2_q = parent_tangents[1].row(q).transpose();
        Eigen::Matrix<T, 3, 1> a3_q = a1_q.cross(a2_q);
        const T a3_norm = a3_q.norm();
        if (a3_norm > T(1e-14)) a3_q /= a3_norm;
        else a3_q = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));

        Eigen::Matrix<T, 3, 1> t_q = tangents[0].row(q).transpose();
        Eigen::Matrix<T, 3, 1> n_vec = sign_n * t_q.cross(a3_q);
        const T n_norm = n_vec.norm();
        if (n_norm > T(1e-14)) n_vec /= n_norm;
        n_mat.row(q) = n_vec.transpose();
    }
    return n_mat;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, 3> BoundaryPatch<T, d>::eval_normal(
    const std::array<ColMatrix<T, 3>, d - 1>& tangents,
    T sign_n,
    const std::array<ColMatrix<T, 3>, d>& /*parent_tangents*/) const requires(d == 3)
{
    const Index Q = tangents[0].rows();
    ColMatrix<T, 3> n_mat(Q, 3);
    for (Index q = 0; q < Q; ++q) {
        Eigen::Matrix<T, 3, 1> t0_q = tangents[0].row(q).transpose();
        Eigen::Matrix<T, 3, 1> t1_q = tangents[1].row(q).transpose();
        Eigen::Matrix<T, 3, 1> n_raw = t0_q.cross(t1_q);
        const T nrm = n_raw.norm();
        n_mat.row(q) = (sign_n * (nrm > T(1e-14) ? n_raw / nrm
                                                  : Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1)))).transpose();
    }
    return n_mat;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
Index BoundaryPatch<T, d>::parent_flat_span(Index boundary_span) const
{
    return parent_span_offset_ + boundary_span * parent_span_stride_;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, d> BoundaryPatch<T, d>::lift_to_parent(const Vector<T>& boundary_pts) const
{
    const Index Q = boundary_pts.size();
    ColMatrix<T, d> pts(Q, d);
    pts.col(param_dim_).setConstant(u_eval_fixed_);
    pts.col(1 - param_dim_) = boundary_pts;
    return pts;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, 3> BoundaryPatch<T, d>::eval_outward_normal(
    Index boundary_span,
    const std::vector<Matrix<T>>& boundary_derivs,
    Index parent_flat_span,
    const std::vector<Matrix<T>>& parent_derivs) const
{
    auto boundary_spans = this->decode_span(boundary_span);
    auto boundary_ctrl_pts = this->active_control_pts(boundary_spans);
    auto boundary_tangents = this->eval_tangent(boundary_derivs, boundary_ctrl_pts);

    auto parent_spans = parent_->decode_span(parent_flat_span);
    auto parent_ctrl_pts = parent_->active_control_pts(parent_spans);
    auto parent_tangents = parent_->eval_tangent(parent_derivs, parent_ctrl_pts);

    return eval_normal(boundary_tangents, sign_n_, parent_tangents);
}

// === Template Instantiations ========================================================

template class BoundaryPatch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BoundaryPatch<float, 2>;
#endif

} // namespace pyck
