#include "patch_boundary.hpp"
#include "patch.hpp"
#include "dof_mapper.hpp"
#include "local_frame.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d> requires (d > 1)
PatchBoundary<T, d>::PatchBoundary(const Ptr<Patch<T, d>>& parent,
                                   std::size_t param_dim,
                                   bool at_start)
    : Patch<T, d - 1>(parent->basis_ptr(1 - param_dim),
                      parent->get_control_points(parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0))),
      parent_(parent), param_dim_(param_dim), at_start_(at_start),
      u_eval_fixed_(T(0)),
      sign_n_(((param_dim == 1) == at_start) ? T(1) : T(-1))
{
    if (param_dim >= d) 
    {
        throw std::invalid_argument("PatchBoundary: "
                                    "param_dim is out of bounds for dimension d."
        );
    }

    parent_dofs_ = parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0);
    Index span_fixed = 0;
    auto& kv_fixed = parent->basis(param_dim).knot_vector();
    
    if (at_start) 
    {
        for (Index s = 0; s < kv_fixed.num_spans(); ++s) 
        {
            auto [lo, hi] = kv_fixed.span_bounds(s);
            if (std::abs(hi - lo) > T(1e-14)) { 
                span_fixed = s; u_eval_fixed_ = lo; 
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
                span_fixed = s; u_eval_fixed_ = hi; 
                break; 
            }
        }
    }

    const Index n_int_u = parent->tensor_product().num_intervals()[0];
    parent_span_offset_ = (param_dim_ == 0) ? span_fixed : span_fixed * n_int_u;
    parent_span_stride_ = (param_dim_ == 0) ? n_int_u : Index(1);
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, d> PatchBoundary<T, d>::lift_to_parent(const Vector<T>& boundary_pts) const requires(d == 2)
{
    const Index Q = boundary_pts.size();
    ColMatrix<T, d> pts(Q, d);
    pts.col(param_dim_).setConstant(u_eval_fixed_);
    pts.col(1 - param_dim_) = boundary_pts;
    return pts;
}

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, 3> 
PatchBoundary<T, d>::eval_outward_normal(const LocalFrame<T, d - 1>& boundary_local,
                                         const LocalFrame<T, d>& parent_local) const requires(d == 2)
{
    const Index Q = boundary_local.a1.rows();
    ColMatrix<T, 3> n_mat(Q, 3);

    for (Index q = 0; q < Q; ++q) 
    {
        // Parent surface normal a_3 = (a_1 × a_2) / ||a_1 × a_2||.
        Eigen::Matrix<T, 3, 1> pa1 = parent_local.a1.row(q).transpose();
        Eigen::Matrix<T, 3, 1> pa2 = parent_local.a2.row(q).transpose();
        Eigen::Matrix<T, 3, 1> a3 = pa1.cross(pa2);
        const T jac_p = parent_local.jac(q);
        if (jac_p > T(1e-14)) a3 /= jac_p;
        else a3 = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));

        // Outward in-surface normal n = sign_n · (a_1^bd × a_3) / ||a_1^bd × a_3||.
        Eigen::Matrix<T, 3, 1> t = boundary_local.a1.row(q).transpose();
        Eigen::Matrix<T, 3, 1> n_vec = sign_n_ * t.cross(a3);
        const T n_norm = n_vec.norm();
        if (n_norm > T(1e-14)) n_vec /= n_norm;
        n_mat.row(q) = n_vec.transpose();
    }
    return n_mat;
}

// === Template Instantiations ========================================================

template class PatchBoundary<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PatchBoundary<float, 2>;
#endif

} // namespace pyck
