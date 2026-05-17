#include "patch.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   const ColMatrix<T, 3>& control_pts) requires(d == 1)
    : control_pts_(control_pts),
      tensor_product_({basis_u}),
      dof_mapper_({tensor_product_.basis(0).num_basis()},
                  {tensor_product_.basis(0).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Control points must be embedded in 3D space.");
    }

    const Index expected_n = this->tensor_product_.basis(0).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Dimension mismatch.");
    }
}

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   Ptr<const Basis<T>> basis_v,
                   const ColMatrix<T, 3>& control_pts) requires(d == 2)
    : control_pts_(control_pts),
      tensor_product_({basis_u, basis_v}),
      dof_mapper_({tensor_product_.basis(0).num_basis(), tensor_product_.basis(1).num_basis()},
                  {tensor_product_.basis(0).degree(),    tensor_product_.basis(1).degree()})
{
    if (control_pts.cols() != 3)
    {
        throw std::invalid_argument("Patch<T, 2>: "
                                    "Control points must be embedded in 3D space.");
    }

    const Index expected_n = this->tensor_product_.basis(0).num_basis()
                           * this->tensor_product_.basis(1).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n)
    {
        throw std::invalid_argument("Patch<T, 2>: "
                                    "Dimension mismatch.");
    }
}

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u,
                   Ptr<const Basis<T>> basis_v,
                   Ptr<const Basis<T>> basis_w,
                   const ColMatrix<T, 3>& control_pts) requires(d == 3)
    : control_pts_(control_pts),
      tensor_product_({basis_u, basis_v, basis_w}),
      dof_mapper_({tensor_product_.basis(0).num_basis(),
                   tensor_product_.basis(1).num_basis(),
                   tensor_product_.basis(2).num_basis()},
                  {tensor_product_.basis(0).degree(),
                   tensor_product_.basis(1).degree(),
                   tensor_product_.basis(2).degree()})
{
    if (control_pts.cols() != 3)
    {
        throw std::invalid_argument("Patch<T, 3>: "
                                    "Control points must be embedded in 3D space.");
    }

    const Index expected_n = this->tensor_product_.basis(0).num_basis()
                           * this->tensor_product_.basis(1).num_basis()
                           * this->tensor_product_.basis(2).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n)
    {
        throw std::invalid_argument("Patch<T, 3>: "
                                    "Dimension mismatch.");
    }
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::active_control_pts(const std::array<Index, d>& spans) const
{
    auto dofs = dof_mapper_.get_element_dofs(spans);
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

    for (Index i = 0; i < indices.size(); ++i) 
    {
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
    if constexpr (d == 1) 
    {
        return {flat_idx};
    } 
    else
    {
        auto intervals = tensor_product_.num_intervals();
        std::array<Index, d> spans;
        Index temp = flat_idx;
        
        for (std::size_t i = 0; i < d; ++i) 
        {
            spans[i] = temp % intervals[i];
            temp /= intervals[i];
        }
        return spans;
    }
}

template <std::floating_point T, std::size_t d>
std::vector<Index>
Patch<T, d>::assembly_dofs() const
{
    std::vector<Index> indices(num_control_pts());
    for (Index i = 0; i < indices.size(); ++i) indices[i] = i;
    return indices;
}

// Helper: evaluate all n basis functions at m points via eval_on_span loop.
// Returns (m × n) dense matrix by scattering span-local results.
template <std::floating_point T>
static Matrix<T> eval_dense_1d(const Basis<T>& basis, const Vector<T>& pts)
{
    const Index n_pts = pts.size();
    const Index n = basis.num_basis();
    const Index p = basis.degree();
    Matrix<T> N = Matrix<T>::Zero(n_pts, n);
    for (Index i = 0; i < n_pts; ++i) {
        const Index span = basis.find_span(pts[i]);
        Vector<T> pt(1);
        pt << pts[i];
        std::vector<Matrix<T>> local;
        basis.eval_on_span(pt, span, 0, local);
        N.row(i).segment(span - p, p + 1) = local[0].row(0);
    }
    return N;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::eval_physical(const Vector<T>& pts) const requires(d == 1)
{
    const Matrix<T> N = eval_dense_1d(tensor_product_.basis(0), pts);
    ColMatrix<T, 3> result(pts.size(), 3);
    result.noalias() = N * control_pts_;
    return result;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::eval_physical(const Matrix<T>& pts) const requires(d == 2)
{
    const Index n_pts = pts.rows();
    const Index n_u = tensor_product_.basis(0).num_basis();
    const Index n_v = tensor_product_.basis(1).num_basis();

    const Vector<T> u_pts = pts.col(0);
    const Vector<T> v_pts = pts.col(1);

    const Matrix<T> N = eval_dense_1d(tensor_product_.basis(0), u_pts);  // (n_pts, n_u)
    const Matrix<T> M = eval_dense_1d(tensor_product_.basis(1), v_pts);  // (n_pts, n_v)

    ColMatrix<T, 3> result(n_pts, 3);

    for (Index k = 0; k < 3; ++k)
    {
        Eigen::Map<const Matrix<T>> CP_k(control_pts_.col(k).data(), n_u, n_v);
        for (Index p = 0; p < n_pts; ++p)
            result(p, k) = (N.row(p) * CP_k * M.row(p).transpose())(0, 0);
    }

    return result;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::eval_physical(const Matrix<T>& pts) const requires(d == 3)
{
    const Index n_pts = pts.rows();
    const Index n_u = tensor_product_.basis(0).num_basis();
    const Index n_v = tensor_product_.basis(1).num_basis();
    const Index n_w = tensor_product_.basis(2).num_basis();

    const Vector<T> u_pts = pts.col(0);
    const Vector<T> v_pts = pts.col(1);
    const Vector<T> w_pts = pts.col(2);

    const Matrix<T> Nu = eval_dense_1d(tensor_product_.basis(0), u_pts); // (n_pts, n_u)
    const Matrix<T> Nv = eval_dense_1d(tensor_product_.basis(1), v_pts); // (n_pts, n_v)
    const Matrix<T> Nw = eval_dense_1d(tensor_product_.basis(2), w_pts); // (n_pts, n_w)

    ColMatrix<T, 3> result(n_pts, 3);

    for (Index dim = 0; dim < 3; ++dim)
    {
        Eigen::Map<const Matrix<T>> CP(control_pts_.col(dim).data(), n_u * n_v, n_w);
        for (Index p = 0; p < n_pts; ++p)
        {
            // Contract over w to get a (n_u·n_v)×1 vector, reshape as (n_u, n_v),
            // then contract over u and v.
            Vector<T> tmp = CP * Nw.row(p).transpose();
            Eigen::Map<const Matrix<T>> tmp_uv(tmp.data(), n_u, n_v);
            result(p, dim) = (Nu.row(p) * tmp_uv * Nv.row(p).transpose())(0, 0);
        }
    }

    return result;
}

template <std::floating_point T, std::size_t d>
Patch<T, d>
Patch<T, d>::insert_knot(std::size_t dir, T u) const
{
    if (dir >= d) {
        throw std::invalid_argument("Patch::insert_knot: direction out of range.");
    }

    auto [basis, transform] = tensor_product_.basis(dir).insert_knot(u);
    const Matrix<T>& Top = transform;

    if constexpr (d == 1)
    {
        ColMatrix<T, 3> new_cps(Top.rows(), 3);
        new_cps.noalias() = Top * control_pts_;
        return Patch<T, d>(basis, new_cps);
    }
    else if constexpr (d == 2)
    {
        const Index n_u = tensor_product_.basis(0).num_basis();
        const Index n_v = tensor_product_.basis(1).num_basis();
        const Index n_u_new = (dir == 0) ? Top.rows() : n_u;
        const Index n_v_new = (dir == 1) ? Top.rows() : n_v;

        ColMatrix<T, 3> new_cps(n_u_new * n_v_new, 3);

        for (Index k = 0; k < 3; ++k)
        {
            Eigen::Map<const Matrix<T>> CP_k(control_pts_.col(k).data(), n_u, n_v);
            Eigen::Map<Matrix<T>> new_k(new_cps.col(k).data(), n_u_new, n_v_new);

            if (dir == 0) {
                new_k.noalias() = Top * CP_k;
            } else {
                new_k.noalias() = CP_k * Top.transpose();
            }
        }

        Ptr<const Basis<T>> basis_u = (dir == 0) ? basis : tensor_product_.basis_ptr(0);
        Ptr<const Basis<T>> basis_v = (dir == 1) ? basis : tensor_product_.basis_ptr(1);
        return Patch<T, d>(basis_u, basis_v, new_cps);
    }
    else  // d == 3
    {
        const Index n_u = tensor_product_.basis(0).num_basis();
        const Index n_v = tensor_product_.basis(1).num_basis();
        const Index n_w = tensor_product_.basis(2).num_basis();
        const Index n_u_new = (dir == 0) ? Top.rows() : n_u;
        const Index n_v_new = (dir == 1) ? Top.rows() : n_v;
        const Index n_w_new = (dir == 2) ? Top.rows() : n_w;

        ColMatrix<T, 3> new_cps(n_u_new * n_v_new * n_w_new, 3);

        for (Index k_coord = 0; k_coord < 3; ++k_coord)
        {
            if (dir == 0) {
                // CP viewed as (n_u, n_v*n_w); transform multiplies on the left.
                Eigen::Map<const Matrix<T>> CP(control_pts_.col(k_coord).data(), n_u, n_v * n_w);
                Eigen::Map<Matrix<T>> NEW(new_cps.col(k_coord).data(), n_u_new, n_v_new * n_w_new);
                NEW.noalias() = Top * CP;
            } else if (dir == 2) {
                // CP viewed as (n_u*n_v, n_w); transform multiplies on the right (transposed).
                Eigen::Map<const Matrix<T>> CP(control_pts_.col(k_coord).data(), n_u * n_v, n_w);
                Eigen::Map<Matrix<T>> NEW(new_cps.col(k_coord).data(), n_u_new * n_v_new, n_w_new);
                NEW.noalias() = CP * Top.transpose();
            } else {
                // dir == 1: per w-slice, view (n_u, n_v) and transform on the right.
                for (Index k_slice = 0; k_slice < n_w; ++k_slice) {
                    Eigen::Map<const Matrix<T>> CP_s(
                        control_pts_.col(k_coord).data() + k_slice * n_u * n_v, n_u, n_v);
                    Eigen::Map<Matrix<T>> NEW_s(
                        new_cps.col(k_coord).data() + k_slice * n_u_new * n_v_new, n_u_new, n_v_new);
                    NEW_s.noalias() = CP_s * Top.transpose();
                }
            }
        }

        Ptr<const Basis<T>> basis_u = (dir == 0) ? basis : tensor_product_.basis_ptr(0);
        Ptr<const Basis<T>> basis_v = (dir == 1) ? basis : tensor_product_.basis_ptr(1);
        Ptr<const Basis<T>> basis_w = (dir == 2) ? basis : tensor_product_.basis_ptr(2);
        return Patch<T, d>(basis_u, basis_v, basis_w, new_cps);
    }
}

template <std::floating_point T, std::size_t d>
Patch<T, d>
Patch<T, d>::elevate_degree(std::size_t dir) const
{
    if (dir >= d) {
        throw std::invalid_argument("Patch::elevate_degree: direction out of range.");
    }

    auto [basis, transform] = tensor_product_.basis(dir).elevate_degree();
    const Matrix<T>& Top = transform;

    if constexpr (d == 1)
    {
        ColMatrix<T, 3> new_cps(Top.rows(), 3);
        new_cps.noalias() = Top * control_pts_;
        return Patch<T, d>(basis, new_cps);
    }
    else if constexpr (d == 2)
    {
        const Index n_u = tensor_product_.basis(0).num_basis();
        const Index n_v = tensor_product_.basis(1).num_basis();
        const Index n_u_new = (dir == 0) ? Top.rows() : n_u;
        const Index n_v_new = (dir == 1) ? Top.rows() : n_v;

        ColMatrix<T, 3> new_cps(n_u_new * n_v_new, 3);

        for (Index k = 0; k < 3; ++k)
        {
            Eigen::Map<const Matrix<T>> CP_k(control_pts_.col(k).data(), n_u, n_v);
            Eigen::Map<Matrix<T>> new_k(new_cps.col(k).data(), n_u_new, n_v_new);

            if (dir == 0) {
                new_k.noalias() = Top * CP_k;
            } else {
                new_k.noalias() = CP_k * Top.transpose();
            }
        }

        Ptr<const Basis<T>> basis_u = (dir == 0) ? basis : tensor_product_.basis_ptr(0);
        Ptr<const Basis<T>> basis_v = (dir == 1) ? basis : tensor_product_.basis_ptr(1);
        return Patch<T, d>(basis_u, basis_v, new_cps);
    }
    else  // d == 3
    {
        const Index n_u = tensor_product_.basis(0).num_basis();
        const Index n_v = tensor_product_.basis(1).num_basis();
        const Index n_w = tensor_product_.basis(2).num_basis();
        const Index n_u_new = (dir == 0) ? Top.rows() : n_u;
        const Index n_v_new = (dir == 1) ? Top.rows() : n_v;
        const Index n_w_new = (dir == 2) ? Top.rows() : n_w;

        ColMatrix<T, 3> new_cps(n_u_new * n_v_new * n_w_new, 3);

        for (Index k_coord = 0; k_coord < 3; ++k_coord)
        {
            if (dir == 0) {
                Eigen::Map<const Matrix<T>> CP(control_pts_.col(k_coord).data(), n_u, n_v * n_w);
                Eigen::Map<Matrix<T>> NEW(new_cps.col(k_coord).data(), n_u_new, n_v_new * n_w_new);
                NEW.noalias() = Top * CP;
            } else if (dir == 2) {
                Eigen::Map<const Matrix<T>> CP(control_pts_.col(k_coord).data(), n_u * n_v, n_w);
                Eigen::Map<Matrix<T>> NEW(new_cps.col(k_coord).data(), n_u_new * n_v_new, n_w_new);
                NEW.noalias() = CP * Top.transpose();
            } else {
                for (Index k_slice = 0; k_slice < n_w; ++k_slice) {
                    Eigen::Map<const Matrix<T>> CP_s(
                        control_pts_.col(k_coord).data() + k_slice * n_u * n_v, n_u, n_v);
                    Eigen::Map<Matrix<T>> NEW_s(
                        new_cps.col(k_coord).data() + k_slice * n_u_new * n_v_new, n_u_new, n_v_new);
                    NEW_s.noalias() = CP_s * Top.transpose();
                }
            }
        }

        Ptr<const Basis<T>> basis_u = (dir == 0) ? basis : tensor_product_.basis_ptr(0);
        Ptr<const Basis<T>> basis_v = (dir == 1) ? basis : tensor_product_.basis_ptr(1);
        Ptr<const Basis<T>> basis_w = (dir == 2) ? basis : tensor_product_.basis_ptr(2);
        return Patch<T, d>(basis_u, basis_v, basis_w, new_cps);
    }
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
