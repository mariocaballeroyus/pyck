#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <vector>
#include <Eigen/Core>

#include "basis.hpp"
#include "tensor.hpp"
#include "dof_mapper.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * Abstract base class for parametric patches defined by the tensor-product of univariate 
 * basis functions.
 */
template <std::floating_point T, std::size_t d>
class Patch
{
public:

    using ScalarType = T;

    // === Constructors ===============================================================

    /// @brief Virtual destructor
    virtual ~Patch() = default;

    /**
     * @brief Construct a new Patch object
     * 
     * @param control_pts The control points of the patch
     */
    explicit Patch(const ColMatrix<T, 3>& control_pts)
        : control_pts_(control_pts) {}

    // === Properties =================================================================

    /// @brief Get the geometric dimension of the patch
    static constexpr std::size_t gdim() 
    { return 3; }

    /// @brief Get the topological dimension of the patch
    static constexpr std::size_t tdim() 
    { return d; }

    /// @brief Get the control points of the patch
    const ColMatrix<T, 3>& control_pts() const 
    { return control_pts_; }

    /// @brief Get the number of control points in the patch
    std::size_t num_control_pts() const 
    { return control_pts_.rows(); }

    /// @brief Get the control points of the patch (non-const version)
    ColMatrix<T, 3>& control_pts() 
    { return control_pts_; }

    /// @brief Get the basis functions for a given parametric direction
    /// @param dir The parametric direction
    virtual const Basis<T>& basis(std::size_t dir) const = 0;

    /// @brief Get the full tensor product basis
    virtual const TensorProduct<T, d>& tensor_product() const = 0;

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the raw basis functions and their parametric derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @param spans  Per-direction knot-span indices.
     * @param order The highest order of derivatives to compute.
     * @return A vector of matrices.  Each matrix has size (Q, K) where K = prod(p_i + 1).
     */
    virtual std::vector<Matrix<T>> eval_basis_functions(const ColMatrix<T, d>& points,
                                                        const std::array<Index, d>& spans,
                                                        std::size_t order = 0) const = 0;
    
    /**
     * @brief Evaluate non-zero shape functions and their covariant derivatives
     *        within a given multi-dimensional knot span, together with the
     *        Jacobian determinant (integration measure) at each point.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @param spans  Per-direction knot-span indices.
     * @param order  The highest order of manifold derivatives to compute.
     * @return A pair of (vector of matrices each (Q, K), Jacobian vector of size Q).
     */          
    virtual std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions(const ColMatrix<T, d>& points,
                         const std::array<Index, d>& spans,
                         std::size_t order = 0) const = 0;

    /**
     * @brief Evaluate the physical geometry mapping and its parametric derivatives
     *        using only the active control points for the given knot span.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @param spans  Per-direction knot-span indices.
     * @param order The highest order of parametric derivatives to compute.
     * @return A vector of matrices, each of size (Q, 3).
     */
    virtual std::vector<ColMatrix<T, 3>> eval_geometry(const ColMatrix<T, d>& points,
                                                       const std::array<Index, d>& spans,
                                                       std::size_t order = 0) const = 0;

    /**
     * @brief Evaluate the Jacobian determinant (integration measure) at each point
     *        within a given knot span.
     *
     * Convenience method — delegates to eval_shape_functions.
     *
     * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @param spans  Per-direction knot-span indices.
     * @return A vector of size (Q) containing the Jacobian determinant at each point.
     */
    virtual Vector<T> eval_jacobian(const ColMatrix<T, d>& points,
                                    const std::array<Index, d>& spans) const
    {
        return eval_shape_functions(points, spans, 0).second;
    }

    // === DOF Mapping ================================================================

    /**
     * @brief Compute the global flattened DOF indices for the clamped boundary layers.
     *        For degree p, a fully clamped boundary requires prescribing the first 2 control points 
     *        (or last 2 control points) along the given parametric dimension.
     * @param param_dim The parametric dimension along which the boundary lies (e.g. 0 for u, 1 for v).
     * @param at_start True to get the first two layers (u=0), False for the last two layers (u=1).
     * @return A vector of flattened global DOF indices belonging to the boundary layers.
     */
    virtual std::vector<Index> boundary_dofs(std::size_t param_dim, bool at_start) const
    { return dof_mapper().get_boundary_dofs(param_dim, at_start); }
    
    /**
     * @brief Get the DofMapper responsible for resolving logical tensor indices 
     *        into global flattened DOF indices for this patch.
     */
    virtual const DofMapper<d>& dof_mapper() const = 0;

    // === Active Control Points ======================================================

    /**
     * @brief Extract the subset of control points that are active (non-zero) in
     *        the given knot span.
     *
     * This uses the DofMapper to determine which global DOF indices correspond
     * to the span, then gathers the matching rows from the control point matrix.
     *
     * @param spans Per-direction knot-span indices.
     * @return A matrix of size (K, 3) where K = prod(p_i + 1) for non-degenerate spans.
     */
    ColMatrix<T, 3> active_control_pts(const std::array<Index, d>& spans) const
    {
        auto dofs = dof_mapper().get_element_dofs(spans);
        ColMatrix<T, 3> pts(dofs.size(), 3);
        for (Index i = 0; i < dofs.size(); ++i) {
            pts.row(i) = control_pts_.row(dofs[i]);
        }
        return pts;
    }

protected:

    // === Member Variables ===========================================================

    /// @brief Control points of the patch, stored as a matrix where each row is a control point
    ColMatrix<T, 3> control_pts_;

};

} // namespace pyck

#endif // PYCK_PATCH_HPP
