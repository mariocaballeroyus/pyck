#include "patch.hpp"

#include <stdexcept>
#include <utility>

#include "evaluation.hpp"

namespace pyck
{

namespace {

/// Apply a 1D linear transform `Top` along parametric direction `dir` of a
/// d-dimensional tensor of control points stored as a (∏ shape_in × 3) column-
/// major matrix.
///
/// View the column-k slice as a (shape_in[0] × shape_in[1] × … × shape_in[d-1])
/// tensor with dim-0-fastest layout (matches DofMapper's `to_global`). The
/// returned matrix has the same layout but with shape_in[dir] replaced by
/// Top.rows().
///
/// `dir == 0`     →  view as (n_dir, suffix), out = Top * view              (BLAS-3)
/// `dir == d-1`   →  view as (prefix, n_dir), out = view * Topᵀ              (BLAS-3)
/// otherwise      →  loop over `suffix` slices, each (prefix, n_dir) * Topᵀ  (BLAS-3 per slice)
///
/// d == 1 falls through the `dir == 0` branch with suffix = 1, matching the
/// `Top * control_pts` fast path for curves.
template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
apply_1d_transform_along_dir(const ColMatrix<T, 3>& cps,
                             const std::array<Index, d>& shape_in,
                             const Matrix<T>& Top,
                             std::size_t dir)
{
    Index prefix = 1, suffix = 1;
    for (std::size_t i = 0;       i < dir; ++i) prefix *= shape_in[i];
    for (std::size_t i = dir + 1; i < d;   ++i) suffix *= shape_in[i];
    const Index n_dir_in  = shape_in[dir];
    const Index n_dir_out = static_cast<Index>(Top.rows());

    ColMatrix<T, 3> out(prefix * n_dir_out * suffix, 3);

    for (Index k = 0; k < 3; ++k)
    {
        const T* src = cps.col(k).data();
        T*       dst = out.col(k).data();

        if (dir == 0) {
            Eigen::Map<const Matrix<T>> M(src, n_dir_in,  suffix);
            Eigen::Map<      Matrix<T>> N(dst, n_dir_out, suffix);
            N.noalias() = Top * M;
        } else if (dir == d - 1) {
            Eigen::Map<const Matrix<T>> M(src, prefix, n_dir_in);
            Eigen::Map<      Matrix<T>> N(dst, prefix, n_dir_out);
            N.noalias() = M * Top.transpose();
        } else {
            // Only reachable when d == 3 and dir == 1.
            for (Index s = 0; s < suffix; ++s) {
                Eigen::Map<const Matrix<T>> M(src + s * prefix * n_dir_in,  prefix, n_dir_in);
                Eigen::Map<      Matrix<T>> N(dst + s * prefix * n_dir_out, prefix, n_dir_out);
                N.noalias() = M * Top.transpose();
            }
        }
    }

    return out;
}

/// Build a `std::array<Index, d>` by applying a member-function `mfn` to each
/// element of a `std::array<Ptr<const Basis<T>>, d>`.
template <std::floating_point T, std::size_t d, typename MemFn>
std::array<Index, d>
basis_attr_array(const std::array<Ptr<const Basis<T>>, d>& bases, MemFn mfn)
{
    std::array<Index, d> out{};
    for (std::size_t i = 0; i < d; ++i) {
        out[i] = (bases[i].get()->*mfn)();
    }
    return out;
}

}  // namespace

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(const std::array<Ptr<const Basis<T>>, d>& bases,
                   const ColMatrix<T, 3>& control_pts)
    : control_pts_(control_pts),
      tensor_product_(bases),
      dof_mapper_(basis_attr_array<T, d>(bases, &Basis<T>::num_basis),
                  basis_attr_array<T, d>(bases, &Basis<T>::degree))
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument(
            "Patch: control points must be embedded in 3D space.");
    }

    Index expected_n = 1;
    for (const auto& b : bases) expected_n *= b->num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch: dimension mismatch.");
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

// Helper: evaluate all n basis functions at m points via eval_on_span loop.
// Returns (m × n) dense matrix by scattering span-local results.
template <std::floating_point T>
static Matrix<T> eval_dense_1d(const Basis<T>& basis, const Vector<T>& pts)
{
    const Index n_pts = pts.size();
    const Index n = basis.num_basis();
    const Index p = basis.degree();
    Matrix<T> N = Matrix<T>::Zero(n_pts, n);
    Evaluator<T> ev;
    std::vector<Matrix<T>> local(1);
    for (Index i = 0; i < n_pts; ++i) {
        const Index span = basis.find_span(pts[i]);
        Vector<T> pt(1);
        pt << pts[i];
        basis.eval_on_span(pt, span, local, ev);
        // local[0] is (p+1) x 1 col-major (N x Q with Q=1); the column holds
        // the N active basis values at the single point `pt`.
        N.row(i).segment(span - p, p + 1) = local[0].col(0).transpose();
    }
    return N;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3>
Patch<T, d>::eval_physical(const ColMatrix<T, d>& pts) const
{
    const Index n_pts = pts.rows();

    std::array<Matrix<T>, d> N;        // N[dir] is (n_pts, num_basis[dir])
    std::array<Index, d>     n_basis;
    for (std::size_t dir = 0; dir < d; ++dir) {
        const Vector<T> pts_dir = pts.col(dir);
        N[dir]       = eval_dense_1d(tensor_product_.basis(dir), pts_dir);
        n_basis[dir] = tensor_product_.basis(dir).num_basis();
    }

    ColMatrix<T, 3> result(n_pts, 3);

    if constexpr (d == 1) {
        // Single 3-column GEMM over all quadrature points and physical coords.
        result.noalias() = N[0] * control_pts_;
    } else {
        // Per-point d-fold sequential contraction, vectorised over the 3 coords.
        // Contract axes d-1, d-2, …, 0; after the last contraction the tensor has
        // shape (1, 3) — the 3 physical coordinates at that point.
        for (Index p = 0; p < n_pts; ++p) {
            ColMatrix<T, 3> tensor = control_pts_;
            Index rem = tensor.rows();
            for (std::size_t step = 0; step < d; ++step) {
                const std::size_t dir = d - 1 - step;
                const Index n_dir  = n_basis[dir];
                const Index prefix = rem / n_dir;
                ColMatrix<T, 3> next(prefix, 3);
                for (Index k = 0; k < 3; ++k) {
                    Eigen::Map<const Matrix<T>> view(
                        tensor.col(k).data(), prefix, n_dir);
                    next.col(k).noalias() = view * N[dir].row(p).transpose();
                }
                tensor = std::move(next);
                rem = prefix;
            }
            result.row(p) = tensor.row(0);
        }
    }
    return result;
}

// Apply a 1D `Top` transform along direction `dir` and rebuild the patch with
// the refined basis substituted in slot `dir`. Used by `insert_knot` and
// `elevate_degree`, which differ only in the source of `Top`.
template <std::floating_point T, std::size_t d>
static Patch<T, d>
refine_along_dir(const Patch<T, d>& self,
                 std::size_t dir,
                 Ptr<const Basis<T>> refined_basis,
                 const Matrix<T>& Top)
{
    std::array<Index, d> shape_in;
    for (std::size_t i = 0; i < d; ++i) {
        shape_in[i] = self.basis(i).num_basis();
    }

    ColMatrix<T, 3> new_cps = apply_1d_transform_along_dir<T, d>(
        self.control_pts(), shape_in, Top, dir);

    std::array<Ptr<const Basis<T>>, d> new_bases;
    for (std::size_t i = 0; i < d; ++i) {
        new_bases[i] = (i == dir) ? refined_basis : self.basis_ptr(i);
    }

    return Patch<T, d>(new_bases, new_cps);
}

template <std::floating_point T, std::size_t d>
Patch<T, d>
Patch<T, d>::insert_knot(std::size_t dir, T u) const
{
    if (dir >= d) {
        throw std::invalid_argument("Patch::insert_knot: direction out of range.");
    }
    auto [basis, transform] = tensor_product_.basis(dir).insert_knot(u);
    return refine_along_dir<T, d>(*this, dir, basis, transform);
}

template <std::floating_point T, std::size_t d>
Patch<T, d>
Patch<T, d>::elevate_degree(std::size_t dir) const
{
    if (dir >= d) {
        throw std::invalid_argument("Patch::elevate_degree: direction out of range.");
    }
    auto [basis, transform] = tensor_product_.basis(dir).elevate_degree();
    return refine_along_dir<T, d>(*this, dir, basis, transform);
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
