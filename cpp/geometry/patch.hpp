#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <tuple>
#include <vector>
#include <memory>
#include <Eigen/Core>

#include "../basis/basis.hpp"
#include "../basis/tensor.hpp"
#include "../basis/bspline.hpp"
#include "dof_mapper.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
class QuadratureRule;


/**
 * @brief Primary template for parametric patches defined by the tensor-product 
 *        of univariate basis functions.
 * 
 * This class provides common data and logic for all patches (curves, surfaces, etc.).
 * Mathematical operations like eval_shape_functions are specialized for each dimension.
 */
/**
 * @brief Primary template for parametric patches.
 */
template <std::floating_point T, std::size_t d>
class Patch
{
public:
    /**
     * @brief 1D Constructor (Curve).
     */
    Patch(Ptr<const Basis<T>> basis_u, 
          const ColMatrix<T, 3>& control_pts) requires(d == 1);

    /**
     * @brief 2D Constructor (Surface).
     */
    Patch(Ptr<const Basis<T>> basis_u,
          Ptr<const Basis<T>> basis_v,
          const ColMatrix<T, 3>& control_pts) requires(d == 2);

    virtual ~Patch() = default;

    std::vector<Matrix<T>> eval_basis_functions(const ColMatrix<T, d>& points,
                                                Index span,
                                                std::size_t order = 0) const;
    
                                                std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions(const ColMatrix<T, d>& points,
                         Index span,
                         std::size_t order = 0) const requires(d == 1);
    std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions(const ColMatrix<T, d>& points,
                         Index span,
                         std::size_t order = 0) const requires(d == 2);

    ColMatrix<T, 3> eval_geometry(const ColMatrix<T, d>& points,
                                  Index span) const requires(d == 1);

    ColMatrix<T, 3> eval_geometry(const ColMatrix<T, d>& points,
                                  Index span) const requires(d == 2);

    ColMatrix<T, 3> eval_physical_points(const QuadratureRule<T, d>& quadrature) const requires(d == 1);
    ColMatrix<T, 3> eval_physical_points(const QuadratureRule<T, d>& quadrature) const requires(d == 2);

    /// @brief Get the Greville points of the patch in the parametric domain.
    const ColMatrix<T, d>& greville_points() const { return greville_points_; }

    /// @brief Get the element span containing the Greville point for a global DOF index.
    Index greville_span(Index dof_index) const { return greville_spans_[dof_index]; }

    /**
     * @brief Evaluate shape functions and their derivatives at the Greville point 
     *        corresponding to a global DOF index.
     * 
     * @param dof_index Global DOF index (0 to num_control_pts - 1).
     * @param order Highest order of derivatives to compute.
     * @return Evaluation results (shape function matrices and area element).
     */
    std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions_at_greville(Index dof_index, std::size_t order = 0) const;

    std::array<ColMatrix<T, 3>, d> eval_tangent(
        const std::vector<Matrix<T>>& N,
        const ColMatrix<T, 3>& act_pts) const requires(d == 1);

    std::array<ColMatrix<T, 3>, d> eval_tangent(
        const std::vector<Matrix<T>>& N,
        const ColMatrix<T, 3>& act_pts) const requires(d == 2);

    /**
     * @brief Evaluate the local covariant frame {a1, a2, a3} at a set of
     *        parametric points within a single span.
     *
     * Returns (a1, a2, a3, jac):
     *   - a1, a2: raw covariant tangents a_α = ∂x/∂ξ^α (Q×3, NOT
     *             unit-normalised; their magnitudes are √g_αα).
     *   - a3:     unit director a3 = (a1×a2)/||a1×a2|| (Q×3).
     *   - jac:    area element ||a1×a2|| = √det(g) (Q).
     *
     * Boundary callers should delegate the surface normal a3 to the parent
     * patch via this method rather than recomputing it locally.
     *
     * @note BoundaryField subclasses currently consume the boundary outward
     * normal as (n_x, n_y) and silently drop n_z. That's fine on flat plates
     * but must be revisited before shipping shell elements where a3 ≠ ẑ.
     */
    std::tuple<ColMatrix<T, 3>, ColMatrix<T, 3>, ColMatrix<T, 3>, Vector<T>>
    eval_local_frame(const ColMatrix<T, d>& points,
                     Index span) const requires(d == 2);

    /// @brief Basis in the given parametric direction.
    const Basis<T>& basis(std::size_t dir) const 
    { return tensor_product_.basis(dir); }

    /// @brief Get a shared pointer to the 1D basis for a given parametric direction
    Ptr<const Basis<T>> basis_ptr(std::size_t dir) const 
    { return tensor_product_.basis_ptr(dir); }

    /// @brief Full tensor-product basis.
    const TensorProduct<T, d>& tensor_product() const 
    { return tensor_product_; }

    /// @brief DOF mapper for this patch.
    const DofMapper<d>& dof_mapper() const 
    { return dof_mapper_; }

    /// @brief Get the geometric dimension (always 3 for now).
    constexpr std::size_t gdim() const { return 3; }

    /// @brief Get the topological dimension.
    constexpr std::size_t tdim() const { return d; }

    /// @brief Get the control points matrix.
    const ColMatrix<T, 3>& control_pts() const { return control_pts_; }
    ColMatrix<T, 3>& control_pts() { return control_pts_; }

    /// @brief Get the number of control points.
    std::size_t num_control_pts() const { return control_pts_.rows(); }

    ColMatrix<T, 3> active_control_pts(const std::array<Index, d>& spans) const;
    ColMatrix<T, 3> get_control_points(const std::vector<Index>& indices) const;
    std::vector<Index> layer_dofs(std::size_t param_dim, bool at_start, Index layer_idx = 0) const;
    std::array<Index, d> decode_span(Index flat_idx) const;

    /// @brief Local indices of this patch's control points (0, 1, …, N−1).
    virtual std::vector<Index> global_indices() const {
        std::vector<Index> indices(num_control_pts());
        for (Index i = 0; i < indices.size(); ++i) indices[i] = i;
        return indices;
    }

    /// @brief DOF indices used for assembly scatter. Defaults to global_indices();
    ///        overridden by PatchBoundary to return the parent patch's DOF indices.
    virtual std::vector<Index> assembly_dofs() const { return global_indices(); }

protected:
    ColMatrix<T, 3> control_pts_;
    TensorProduct<T, d> tensor_product_;
    DofMapper<d> dof_mapper_;
    ColMatrix<T, d> greville_points_;
    std::vector<Index> greville_spans_;
};

template <std::floating_point T, std::size_t d>
std::vector<Matrix<T>> Patch<T, d>::eval_basis_functions(const ColMatrix<T, d>& points, Index span, std::size_t order) const
{
    auto spans = decode_span(span);
    return tensor_product_.eval_on_span(points, spans, static_cast<Index>(order));
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::active_control_pts(const std::array<Index, d>& spans) const
{
    auto dofs = dof_mapper_.get_element_dofs(spans);
    ColMatrix<T, 3> pts(dofs.size(), 3);
    for (Index i = 0; i < dofs.size(); ++i) {
        pts.row(i) = control_pts_.row(dofs[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::get_control_points(const std::vector<Index>& indices) const
{
    ColMatrix<T, 3> pts(indices.size(), 3);
    for (Index i = 0; i < indices.size(); ++i) {
        pts.row(i) = control_pts_.row(indices[i]);
    }
    return pts;
}

template <std::floating_point T, std::size_t d>
std::vector<Index> Patch<T, d>::layer_dofs(std::size_t param_dim, bool at_start, Index layer_idx) const
{
    return dof_mapper_.get_layer_dofs(param_dim, at_start, layer_idx);
}

template <std::floating_point T, std::size_t d>
std::array<Index, d> Patch<T, d>::decode_span(Index flat_idx) const
{
    if constexpr (d == 1) {
        return {flat_idx};
    } else {
        auto intervals = tensor_product_.num_intervals();
        std::array<Index, d> spans;
        Index temp = flat_idx;
        for (std::size_t i = 0; i < d; ++i) {
            spans[i] = temp % intervals[i];
            temp /= intervals[i];
        }
        return spans;
    }
}

} // namespace pyck

#endif // PYCK_PATCH_HPP
