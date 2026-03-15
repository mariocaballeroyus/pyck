#ifndef PYCK_SURFACE_HPP
#define PYCK_SURFACE_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "basis.hpp"
#include "patch.hpp"
#include "tensor.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief A parametric surface patch defined by two univariate bases (tensor product),
 *        embedded in a 3-dimensional space.
 * 
 * The surface is parameterised by (u, v) ∈ [0,1]² and maps to 3D via:
 *   x(u,v) = Σ_A N^A(u,v) P_A
 *
 * Physical (covariant) derivatives are computed using the surface metric
 * tensor and Christoffel symbols of the second kind, generalising the
 * arc-length push-forward used by CurvePatch.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class SurfacePatch : public Patch<T, 2>
{
public:

    /**
     * @brief Construct a surface patch from two 1D bases and a control-point grid.
     *
     * @param basis_u   Basis in the u (first) parametric direction.
     * @param basis_v   Basis in the v (second) parametric direction.
     * @param control_pts  Control-point matrix of size (n_u * n_v, 3), stored
     *                     in u-fastest (column-major logical) order.
     */
    SurfacePatch(Ptr<const Basis<T>> basis_u,
                 Ptr<const Basis<T>> basis_v,
                 const ColMatrix<T, 3>& control_pts);

    /**
     * @brief Evaluate the non-zero raw basis functions and their parametric
     *        derivatives within a given element (flat knot-span index).
     *
     * @param points  Evaluation points in parametric coordinates (Q × 2).
     * @param span    Flat element index.
     * @param order   Highest derivative order.
     * @return Vector of (order+1)² matrices, each (Q, K), following the
     *         tensor-product multi-index layout produced by TensorProduct.
     */
    std::vector<Matrix<T>> eval_basis_functions(const ColMatrix<T, 2>& points,
                                                Index span,
                                                std::size_t order = 0) const override;

    /**
     * @brief Evaluate non-zero shape functions and their physical (covariant)
     *        derivatives, together with the Jacobian determinant (area element).
     *
     * The output layout is:
     *   - result[0]  = N           (values)
     *   - result[1]  = N_{,u}      (first parametric derivative in u)
     *   - result[2]  = N_{,v}      (first parametric derivative in v)
     *   - result[3]  = N_{;uu}     (second covariant derivative)
     *   - result[4]  = N_{;uv}     (mixed second covariant derivative)
     *   - result[5]  = N_{;vv}     (second covariant derivative)
     *
     * where the second covariant derivative includes the Christoffel correction:
     *   N_{;αβ} = N_{,αβ} − Γ^γ_{αβ} N_{,γ}
     *
     * @param points  Evaluation points in parametric coordinates (Q × 2).
     * @param span    Flat element index.
     * @param order   Highest derivative order (clamped to [1, 2]).
     * @return Pair of (derivative matrices, Jacobian vector of size Q).
     */
    std::pair<std::vector<Matrix<T>>, Vector<T>>
    eval_shape_functions(const ColMatrix<T, 2>& points,
                         Index span,
                         std::size_t order = 0) const override;

    /**
     * @brief Evaluate physical surface coordinates at parametric points
     *        using the tensor-product basis (no derivatives).
     *
     * @param points  Evaluation points in parametric coordinates (Q × 2).
     * @param span    Flat element index.
     * @return Matrix (Q, 3) of physical coordinates.
     */
    ColMatrix<T, 3> eval_geometry(const ColMatrix<T, 2>& points,
                                  Index span) const override;

    /**
     * @brief Evaluate physical coordinates of all active quadrature points.
     *
     * @param quadrature  2D quadrature rule.
     * @return Matrix of physical coordinates (n_active × Q, 3).
     */
    ColMatrix<T, 3> eval_physical_points(const QuadratureRule<T, 2>& quadrature) const override;

    /// @brief Basis in the given parametric direction (0 = u, 1 = v).
    const Basis<T>& basis(std::size_t dir) const override
    { return tensor_product_.basis(dir); }

    /// @brief Full tensor-product basis.
    const TensorProduct<T, 2>& tensor_product() const override
    { return tensor_product_; }

    /// @brief DOF mapper for this patch.
    const DofMapper<2>& dof_mapper() const override
    { return dof_mapper_; }

private:

    /// @brief Decode a flat element index into per-direction span indices.
    std::array<Index, 2> decode_span(Index flat_idx) const;

    /// @brief Tensor product of the two 1D bases.
    TensorProduct<T, 2> tensor_product_;

    /// @brief DOF mapper for flattened tensor-product indices.
    DofMapper<2> dof_mapper_;
};


// === Free-standing factory functions ================================================

/**
 * @brief Create a flat rectangular surface patch in the xy-plane.
 *
 * Control points are placed at the tensor-product Greville abscissae
 * so that the bilinear map x(u,v) = (width·u, height·v, 0) is exact
 * for any polynomial degree.
 *
 * @param basis_u  Basis in the u direction.
 * @param basis_v  Basis in the v direction.
 * @param width    Physical width  (x-extent).
 * @param height   Physical height (y-extent).
 */
template <std::floating_point T>
SurfacePatch<T> rectangle(Ptr<const Basis<T>> basis_u,
                           Ptr<const Basis<T>> basis_v,
                           T width,
                           T height);

} // namespace pyck

#endif // PYCK_SURFACE_HPP
