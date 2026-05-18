#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <tuple>
#include <type_traits>
#include <vector>
#include <memory>
#include <Eigen/Core>

#include "basis.hpp"
#include "tensor.hpp"
#include "bspline.hpp"
#include "basis_derivs.hpp"
#include "dof_mapper.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
class QuadratureRule;

/**
 * @brief Parametric patch defined by the tensor-product of univariate basis functions.
 */
template <std::floating_point T, std::size_t d>
class Patch
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Primary constructor: build from `d` univariate bases and a
     *        control-point matrix in 3D physical space.
     *
     * The row count of `control_pts` must match ∏ basesᵢ->num_basis(), and
     * it must have 3 columns. All other ctors below are one-line forwarders
     * to this single algorithm.
     */
    Patch(const std::array<Ptr<const Basis<T>>, d>& bases,
          const ColMatrix<T, 3>& control_pts);

    /// @brief Ergonomic forwarder for curve patches (d = 1).
    Patch(Ptr<const Basis<T>> basis_u,
          const ColMatrix<T, 3>& control_pts) requires (d == 1)
        : Patch(std::array<Ptr<const Basis<T>>, 1>{basis_u}, control_pts) {}

    /// @brief Ergonomic forwarder for surface patches (d = 2).
    Patch(Ptr<const Basis<T>> basis_u,
          Ptr<const Basis<T>> basis_v,
          const ColMatrix<T, 3>& control_pts) requires (d == 2)
        : Patch(std::array<Ptr<const Basis<T>>, 2>{basis_u, basis_v}, control_pts) {}

    /// @brief Ergonomic forwarder for volume patches (d = 3).
    Patch(Ptr<const Basis<T>> basis_u,
          Ptr<const Basis<T>> basis_v,
          Ptr<const Basis<T>> basis_w,
          const ColMatrix<T, 3>& control_pts) requires (d == 3)
        : Patch(std::array<Ptr<const Basis<T>>, 3>{basis_u, basis_v, basis_w},
                control_pts) {}

    /**
     * @brief Virtual destructor.
     */
    virtual ~Patch() = default;

    // === Getters ====================================================================

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
    constexpr std::size_t gdim() const 
    { return 3; }

    /// @brief Get the topological dimension.
    constexpr std::size_t tdim() const 
    { return d; }

    /// @brief Get the control points matrix (const).
    const ColMatrix<T, 3>& control_pts() const 
    { return control_pts_; }

    /// @brief Get the control points matrix (non-const).
    ColMatrix<T, 3>& control_pts() 
    { return control_pts_; }

    /// @brief Get the number of control points.
    std::size_t num_control_pts() const 
    { return control_pts_.rows(); }

    ColMatrix<T, 3> active_control_pts(const std::array<Index, d>& spans) const;

    /// @brief Convenience overload: flat span index. Decodes internally.
    ColMatrix<T, 3> active_control_pts(Index span_idx) const
    { return this->active_control_pts(this->decode_span(span_idx)); }

    /// @brief Get the DOF indices used for assembly scatter. Defaults to the
    ///        local indices (0, 1, …, N−1); overridden by PatchBoundary.
    virtual std::vector<Index> assembly_dofs() const;

    // === Utility Functions ==========================================================

    /**
     * @brief Get the control points for a given vector of indices.
     *
     * @param indices Vector of control point indices.
     * @return Matrix containing the control points.
     */
    ColMatrix<T, 3> get_control_points(const std::vector<Index>& indices) const;

    /**
     * @brief Get the layer DOFs for a given parameter dimension and layer index.
     *
     * @param param_dim Parameter dimension.
     * @param at_start Whether the layer is at the start of the parameter dimension.
     * @param layer_idx Layer index.
     * @return Vector of DOF indices.
     */
    std::vector<Index> layer_dofs(std::size_t param_dim,
                                  bool at_start,
                                  Index layer_idx = 0) const;

    /**
     * @brief Decode a flat span index into an array of span indices.
     *
     * @param flat_idx Flat span index.
     * @return Array of span indices.
     */
    std::array<Index, d> decode_span(Index flat_idx) const;

    // === Physical Evaluation ========================================================

    /**
     * @brief Evaluate physical coordinates at parametric sample points.
     *
     * @param pts Parametric sample points, shape (n_pts, d). For d=1 this is a
     *            column vector; for d=2 columns are (u, v); for d=3 (u, v, w).
     * @return Physical coordinates, shape (n_pts, 3).
     */
    ColMatrix<T, 3> eval_physical(const ColMatrix<T, d>& pts) const;

    // === Refinement =================================================================

    /**
     * @brief Refine the patch by inserting a knot value `u` in direction `dir`.
     *
     * The returned patch is geometrically identical to this one (within rounding)
     * but has a refined basis and updated control points. The original patch is
     * unmodified. Derived objects (boundaries, assembler entries, conditions) must
     * be rebuilt against the returned patch.
     *
     * @param dir Parametric direction (0 for d=1; 0 or 1 for d=2).
     * @param u   Knot value to insert.
     * @return A new Patch with the refined basis and control points. Callers
     *         wanting multiplicity > 1 chain the method.
     */
    Patch<T, d> insert_knot(std::size_t dir, T u) const;

    /// @brief Convenience overload for 1D patches: direction is always 0.
    Patch<T, d> insert_knot(T u) const requires(d == 1)
    { return this->insert_knot(0, u); }

    /**
     * @brief Refine the patch by degree-elevating in direction `dir` by one.
     *
     * Returns a new patch with the elevated basis (degree increased by one)
     * and updated control points. Continuity at every existing internal knot
     * is preserved. Callers wanting to elevate by more than one chain the
     * method.
     */
    Patch<T, d> elevate_degree(std::size_t dir) const;

    /// @brief Convenience overload for 1D patches: direction is always 0.
    Patch<T, d> elevate_degree() const requires(d == 1)
    { return this->elevate_degree(0); }

protected:

    /// @brief Patch control point coordinates in the 3-dimensional Euclidean space.
    ColMatrix<T, 3> control_pts_;

    /// @brief Tensor product of univariate basis functions defining the patch geometry.
    TensorProduct<T, d> tensor_product_;

    /// @brief Map from global DOF indices to local indices on this patch.
    DofMapper<d> dof_mapper_;
};

/**
 * @brief Evaluate basis functions and derivatives at the given coordinates
 *        within one element span of a patch (ergonomic Patch-level overload).
 *
 * @param patch       Source patch.
 * @param eval_coords (Q × d) parametric coordinates on the span.
 * @param span_idx    Flat element / span index; decoded internally.
 * @param order       Highest total derivative order (0, 1, 2 or 3).
 */
template <std::floating_point T, std::size_t d>
inline BasisDerivs<T, d>
eval_basis(const Patch<T, d>& patch,
           const std::type_identity_t<ColMatrix<T, d>>& eval_coords,
           Index span_idx,
           std::size_t order = 0)
{
    return eval_basis(patch.tensor_product(),
                      eval_coords,
                      patch.decode_span(span_idx),
                      static_cast<Index>(order));
}

} // namespace pyck

#endif // PYCK_PATCH_HPP
