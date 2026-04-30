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
      parent_(parent), param_dim_(param_dim), at_start_(at_start)
{
    if (param_dim >= d) {
        throw std::invalid_argument(
            "BoundaryPatch: param_dim is out of bounds for dimension d."
        );
    }

    parent_dofs_ = parent->dof_mapper().get_layer_dofs(param_dim, at_start, 0);
}

// === eval_normal ====================================================================

template <std::floating_point T, std::size_t d> requires (d > 1)
ColMatrix<T, 3> BoundaryPatch<T, d>::eval_normal(
    const std::array<ColMatrix<T, 3>, d - 1>& tangents,
    T sign_n,
    const std::array<Eigen::Matrix<T, 3, 1>, d>& parent_tangents) const
{
    const Index Q = tangents[0].rows();
    ColMatrix<T, 3> n_mat(Q, 3);

    if constexpr (d == 2) {
        // 1D boundary on a 2D surface: outward normal = tangent × surface_normal
        Eigen::Matrix<T, 3, 1> a3 = parent_tangents[0].cross(parent_tangents[1]);
        const T a3_norm = a3.norm();
        if (a3_norm > T(1e-14)) a3 /= a3_norm;
        else a3 = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));

        for (Index q = 0; q < Q; ++q) {
            Eigen::Matrix<T, 3, 1> t_q = tangents[0].row(q).transpose();
            Eigen::Matrix<T, 3, 1> n_vec = sign_n * t_q.cross(a3);
            const T n_norm = n_vec.norm();
            if (n_norm > T(1e-14)) n_vec /= n_norm;
            n_mat.row(q) = n_vec.transpose();
        }
    } else if constexpr (d == 3) {
        // 2D boundary on a 3D volume: outward normal = tangent0 × tangent1
        for (Index q = 0; q < Q; ++q) {
            Eigen::Matrix<T, 3, 1> t0_q = tangents[0].row(q).transpose();
            Eigen::Matrix<T, 3, 1> t1_q = tangents[1].row(q).transpose();
            Eigen::Matrix<T, 3, 1> n_raw = t0_q.cross(t1_q);
            const T nrm = n_raw.norm();
            n_mat.row(q) = (sign_n * (nrm > T(1e-14) ? n_raw / nrm
                                                      : Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1)))).transpose();
        }
    }

    return n_mat;
}

// === Template Instantiations ========================================================

template class BoundaryPatch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BoundaryPatch<float, 2>;
#endif

} // namespace pyck
