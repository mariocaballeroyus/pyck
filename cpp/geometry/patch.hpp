#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <vector>
#include <Eigen/Core>

#include "basis.hpp"
#include "tensor.hpp"
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

    explicit Patch(const ColMatrix<T, 3>& control_pts)
    : control_pts_(control_pts) 
    {}

    // === Properties =================================================================

    /// @brief Get the geometric dimension of the patch
    static constexpr std::size_t gdim() { return 3; }

    /// @brief Get the topological dimension of the patch
    static constexpr std::size_t tdim() { return d; }

    /// @brief Get the control points of the patch
    const ColMatrix<T, 3>& control_pts() const { return control_pts_; }

    /// @brief Get the number of control points in the patch
    std::size_t num_control_pts() const { return control_pts_.rows(); }

    /// @brief Get the control points of the patch (non-const version)
    ColMatrix<T, 3>& control_pts() { return control_pts_; }

    /// @brief Get the basis functions for a given parametric direction
    /// @param dir The parametric direction
    virtual const Basis<T>& basis(std::size_t dir) const = 0;

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the raw basis functions and their parametric derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @param order The highest order of derivatives to compute.
     * @return A vector of matrices ordered lexicographically by derivative degree.
     *         The size of each matrix is (Q, K) where K is the number of basis functions.
     * 
     *      Let d=2 (surface patch geometry) and let parametric coordinates be (u,v):
     * 
     *      n = 0 : (0,0) -> [N]        (Basis functions)
     *      n = 1 : (1,0) -> [dN/du]
     *      n = 2 : (0,1) -> [dN/dv]
     *      n = 3 : (2,0) -> [d2N/du2]
     *      n = 4 : (1,1) -> [d2N/dudv]
     *      n = 5 : (0,2) -> [d2N/dv2]
     *      ...
     */
    virtual std::vector<Matrix<T>> eval_basis_functions(const ColMatrix<T, d>& points, 
                                                        std::size_t order = 0) const = 0;
    
    /**
     * @brief Evaluate shape functions and their covariant derivatives expressed in the local 
     *        orthonormal basis of the tangent space at each evaluation point.
     * 
     *        The derivatives are obtained by mapping parametric derivatives using the surface 
     *        Jacobian and Christoffel symbols to form covariant derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @param order  The highest order of manifold derivatives to compute.
     * @return A vector of matrices ordered lexicographically by derivative degree.
     *         The size of each matrix is (Q, K) where K is the number of basis functions.
     * 
     *      Let d=2 (surface patch geometry) and let local tangent coordinates be (x,y):
     * 
     *      n = 0 : (0,0) -> [N]        (Shape functions)
     *      n = 1 : (1,0) -> [dN/dx]
     *      n = 2 : (0,1) -> [dN/dy]      
     *      n = 3 : (2,0) -> [d2N/dx2]
     *      n = 4 : (1,1) -> [d2N/dxdy]
     *      n = 5 : (0,2) -> [d2N/dy2]
     *      ...
     */          
    virtual std::vector<Matrix<T>> eval_shape_functions(const ColMatrix<T, d>& points,
                                                        std::size_t order = 0) const = 0;

    /**
     * @brief Evaluate the physical geometry mapping and its parametric derivatives.
     *        The geometry is computeed as the dot product of the tensor-product basis functions 
     *        and the control points as X(u) = N_i(u) P_i.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @param order The highest order of parametric derivatives to compute.
     * @return A vector of matrices ordered lexicographically by derivative degree.
     *         The size of each matrix is (Q, 3).
     * 
     *      Let d=2 (surface patch geometry) and let parametric coordinates be (u,v):
     * 
     *      n = 0 : (0,0) -> [X]       (Physical coordinates / Position)
     *      n = 1 : (1,0) -> [dX/du]
     *      n = 2 : (0,1) -> [dX/dv]
     *      n = 3 : (2,0) -> [d2X/du2]
     *      n = 4 : (1,1) -> [d2X/dudv]
     *      n = 5 : (0,2) -> [d2X/dv2]
     *      ...
     */
    virtual std::vector<ColMatrix<T, 3>> eval_geometry(const ColMatrix<T, d>& points, 
                                                       std::size_t order = 0) const = 0;

    /**
     * @brief Evaluate the Jacobian determinant (integration measure) at each point.
     * * The Jacobian determinant represents the ratio of the physical "volume" 
     * (length, area, or volume) to the parametric "volume".
     * * - If d = 1: Returns the arc-length increment ||dX/du||.
     * - If d = 2: Returns the area increment ||dX/du × dX/dv||.
     * - If d = 3: Returns the volume increment |det(J)|.
     * * @param points Evaluation points in parametric coordinates as a (Q × d) matrix.
     * @return A vector of size (Q) containing the Jacobian determinant at each point.
     */
    virtual Vector<T> eval_jacobian(const ColMatrix<T, d>& points) const = 0;

    /**
     * @brief Compute the global flattened DOF indices for the clamped boundary layers.
     *        For degree p, a fully clamped boundary requires prescribing the first 2 control points 
     *        (or last 2 control points) along the given parametric dimension.
     * @param param_dim The parametric dimension along which the boundary lies (e.g. 0 for u, 1 for v).
     * @param at_start True to get the first two layers (u=0), False for the last two layers (u=1).
     * @return A vector of flattened global DOF indices belonging to the boundary layers.
     */
    virtual std::vector<Index> get_boundary_dofs(std::size_t param_dim, bool at_start) const = 0;

protected:

    // === Member Variables ===========================================================

    /// @brief Control points of the patch, stored as a matrix where each row is a control point
    ColMatrix<T, 3> control_pts_;

};

} // namespace pyck

#endif // PYCK_PATCH_HPP
