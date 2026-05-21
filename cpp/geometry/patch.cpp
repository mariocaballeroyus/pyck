#include "patch.hpp"

#include <stdexcept>
#include <utility>

#include "evaluation.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   const ColMatrix<T, 3>& control_pts) requires (d == 1)
    : control_pts_(control_pts),
      tensor_product_(std::array<Ptr<const Basis<T>>, 1>{basis_u}),
      dof_mapper_(std::array<Index, 1>{basis_u->num_basis()},
                  std::array<Index, 1>{basis_u->degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument(
            "Patch: control points must be embedded in 3D space.");
    }
    if (static_cast<Index>(control_pts.rows()) != basis_u->num_basis()) {
        throw std::invalid_argument("Patch: dimension mismatch.");
    }
}

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   Ptr<const Basis<T>> basis_v,
                   const ColMatrix<T, 3>& control_pts) requires (d == 2)
    : control_pts_(control_pts),
      tensor_product_(std::array<Ptr<const Basis<T>>, 2>{basis_u, basis_v}),
      dof_mapper_(std::array<Index, 2>{basis_u->num_basis(), basis_v->num_basis()},
                  std::array<Index, 2>{basis_u->degree(),    basis_v->degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument(
            "Patch: control points must be embedded in 3D space.");
    }
    const Index expected = basis_u->num_basis() * basis_v->num_basis();
    if (static_cast<Index>(control_pts.rows()) != expected) {
        throw std::invalid_argument("Patch: dimension mismatch.");
    }
}

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   Ptr<const Basis<T>> basis_v,
                   Ptr<const Basis<T>> basis_w,
                   const ColMatrix<T, 3>& control_pts) requires (d == 3)
    : control_pts_(control_pts),
      tensor_product_(std::array<Ptr<const Basis<T>>, 3>{basis_u, basis_v, basis_w}),
      dof_mapper_(std::array<Index, 3>{basis_u->num_basis(), basis_v->num_basis(), basis_w->num_basis()},
                  std::array<Index, 3>{basis_u->degree(),    basis_v->degree(),    basis_w->degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument(
            "Patch: control points must be embedded in 3D space.");
    }
    const Index expected =
        basis_u->num_basis() * basis_v->num_basis() * basis_w->num_basis();
    if (static_cast<Index>(control_pts.rows()) != expected) {
        throw std::invalid_argument("Patch: dimension mismatch.");
    }
}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::active_control_pts(const std::array<Index, d>& spans) const
{
    std::vector<Index> dofs;
    dof_mapper_.get_element_dofs(spans, dofs);
    ColMatrix<T, 3> pts(dofs.size(), 3);
    for (Index i = 0; i < dofs.size(); ++i) {
        pts.row(i) = control_pts_.row(dofs[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::get_control_points(const std::vector<Index>& indices) const
{
    ColMatrix<T, 3> pts(indices.size(), 3);

    for (Index i = 0; i < indices.size(); ++i) {
        pts.row(i) = control_pts_.row(indices[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
std::vector<Index>
Patch<T, d>::layer_dofs(std::size_t param_dim, bool at_start, Index layer_idx) const
{
    return dof_mapper_.get_layer_dofs(param_dim, at_start, layer_idx);
}

template <std::floating_point T, std::size_t d>
std::array<Index, d>
Patch<T, d>::decode_span(Index flat_idx) const
{
    const auto intervals = tensor_product_.num_intervals();
    std::array<Index, d> spans;
    Index temp = flat_idx;
    for (std::size_t i = 0; i < d; ++i) {
        spans[i] = temp % intervals[i];
        temp /= intervals[i];
    }
    return spans;
}

template <std::floating_point T, std::size_t d>
std::vector<Index>
Patch<T, d>::assembly_dofs() const
{
    std::vector<Index> indices(num_control_pts());
    for (Index i = 0; i < indices.size(); ++i) indices[i] = i;
    return indices;
}

// === Refinement =====================================================================

template <std::floating_point T, std::size_t d>
static Patch<T, d> transform_patch_basis(const Patch<T, d>& self,
                                        std::size_t dir,
                                        Ptr<const Basis<T>> new_basis,
                                        const Matrix<T>& transformation)
{
    std::array<Index, d> shape_in;
    for (std::size_t i = 0; i < d; ++i) {
        shape_in[i] = self.basis(i).num_basis();
    }

    Index prefix = 1, suffix = 1;
    for (std::size_t i = 0;       i < dir; ++i) prefix *= shape_in[i];
    for (std::size_t i = dir + 1; i < d;   ++i) suffix *= shape_in[i];
    const Index n_dir_in  = shape_in[dir];
    const Index n_dir_out = static_cast<Index>(transformation.rows());

    const ColMatrix<T, 3>& cps = self.control_pts();
    ColMatrix<T, 3> new_cps(prefix * n_dir_out * suffix, 3);

    for (Index k = 0; k < 3; ++k) {
        const T* src = cps.col(k).data();
        T*       dst = new_cps.col(k).data();

        if (dir == 0) {
            Eigen::Map<const Matrix<T>> M(src, n_dir_in,  suffix);
            Eigen::Map<      Matrix<T>> N(dst, n_dir_out, suffix);
            N.noalias() = transformation * M;
        } else if (dir == d - 1) {
            Eigen::Map<const Matrix<T>> M(src, prefix, n_dir_in);
            Eigen::Map<      Matrix<T>> N(dst, prefix, n_dir_out);
            N.noalias() = M * transformation.transpose();
        } else {
            // Only reachable when d == 3 and dir == 1.
            for (Index s = 0; s < suffix; ++s) {
                Eigen::Map<const Matrix<T>> M(src + s * prefix * n_dir_in,  prefix, n_dir_in);
                Eigen::Map<      Matrix<T>> N(dst + s * prefix * n_dir_out, prefix, n_dir_out);
                N.noalias() = M * transformation.transpose();
            }
        }
    }

    std::array<Ptr<const Basis<T>>, d> new_bases;
    for (std::size_t i = 0; i < d; ++i) {
        new_bases[i] = (i == dir) ? new_basis : self.basis_ptr(i);
    }

    if constexpr (d == 1) {
        return Patch<T, 1>(new_bases[0], new_cps);
    } else if constexpr (d == 2) {
        return Patch<T, 2>(new_bases[0], new_bases[1], new_cps);
    } else {
        return Patch<T, 3>(new_bases[0], new_bases[1], new_bases[2], new_cps);
    }
}

template <std::floating_point T, std::size_t d>
Patch<T, d>
Patch<T, d>::insert_knot(std::size_t dir, T u) const
{
    if (dir >= d) {
        throw std::invalid_argument("Patch::insert_knot: "
                                    "direction out of range.");
    }
    auto [basis, transformation] = tensor_product_.basis(dir).insert_knot(u);
    return transform_patch_basis<T, d>(*this, dir, basis, transformation);
}

template <std::floating_point T, std::size_t d>
Patch<T, d>
Patch<T, d>::elevate_degree(std::size_t dir) const
{
    if (dir >= d) {
        throw std::invalid_argument("Patch::elevate_degree: "
                                    "direction out of range.");
    }
    auto [basis, transformation] = tensor_product_.basis(dir).elevate_degree();
    return transform_patch_basis<T, d>(*this, dir, basis, transformation);
}

// === Template Specializations =======================================================

template class Patch<double, 1>;
template class Patch<double, 2>;
template class Patch<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 1>;
template class Patch<float, 2>;
template class Patch<float, 3>;
#endif

} // namespace pyck
