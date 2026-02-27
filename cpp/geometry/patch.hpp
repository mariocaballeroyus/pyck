#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <vector>
#include <Eigen/Core>

#include "basis.hpp"
#include "tensor.hpp"

namespace pyck
{

/**
 * Abstract base class for parametric patches defined by the tensor-product of univariate 
 * basis functions.
 */
class Patch
{
public:

    // === Constructors ===============================================================

    /// @brief Virtual destructor
    virtual ~Patch() = default;

    Patch(std::size_t gdim, 
          std::size_t tdim, 
          const Eigen::MatrixXd& control_pts)
    : gdim_(gdim), 
      tdim_(tdim), 
      control_pts_(control_pts) 
    {}

    // === Properties =================================================================

    /// @brief Get the geometric dimension of the patch
    std::size_t gdim() const { return gdim_; }

    /// @brief Get the topological dimension of the patch
    std::size_t tdim() const { return tdim_; }

    /// @brief Get the control points of the patch
    const Eigen::MatrixXd& control_pts() const { return control_pts_; }

    /// @brief Get the number of control points in the patch
    std::size_t num_control_pts() const { return control_pts_.rows(); }

    /// @brief Get the control points of the patch (non-const version)
    Eigen::MatrixXd& control_pts() { return control_pts_; }

    /// @brief Get the basis functions for a given parametric direction
    /// @param dir The parametric direction
    virtual const pyck::Basis& basis(std::size_t dir) const = 0;

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the raw basis functions and their parametric derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × tdim) matrix.
     * @param order The highest order of derivatives to compute.
     * @return A vector of matrices ordered lexicographically by derivative degree.
     *         The size of each matrix is (Q, K) where K is the number of basis functions.
     * 
     *      Let tdim=2 (surface patch geometry) and let parametric coordinates be (u,v):
     * 
     *      n = 0 : (0,0) -> [N]        (Basis functions)
     *      n = 1 : (1,0) -> [dN/du]
     *      n = 2 : (0,1) -> [dN/dv]
     *      n = 3 : (2,0) -> [d2N/du2]
     *      n = 4 : (1,1) -> [d2N/dudv]
     *      n = 5 : (0,2) -> [d2N/dv2]
     *      ...
     */
    virtual std::vector<Eigen::MatrixXd> eval_basis_functions(const Eigen::MatrixXd& points, 
                                                              std::size_t order = 0) const = 0;
    
    /**
     * @brief Evaluate shape functions and their covariant derivatives expressed in the local 
     *        orthonormal basis of the tangent space at each evaluation point.
     * 
     *        The derivatives are obtained by mapping parametric derivatives using the surface 
     *        Jacobian and Christoffel symbols to form covariant derivatives.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × tdim) matrix.
     * @param order  The highest order of manifold derivatives to compute.
     * @return A vector of matrices ordered lexicographically by derivative degree.
     *         The size of each matrix is (Q, K) where K is the number of basis functions.
     * 
     *      Let tdim=2 (surface patch geometry) and let local tangent coordinates be (x,y):
     * 
     *      n = 0 : (0,0) -> [N]        (Shape functions)
     *      n = 1 : (1,0) -> [dN/dx]
     *      n = 2 : (0,1) -> [dN/dy]      
     *      n = 3 : (2,0) -> [d2N/dx2]
     *      n = 4 : (1,1) -> [d2N/dxdy]
     *      n = 5 : (0,2) -> [d2N/dy2]
     *      ...
     */          
    virtual std::vector<Eigen::MatrixXd> eval_shape_functions(const Eigen::MatrixXd& points,
                                                              std::size_t order = 0) const = 0;

    /**
     * @brief Evaluate the physical geometry mapping and its parametric derivatives.
     *        The geometry is computeed as the dot product of the tensor-product basis functions 
     *        and the control points as X(u) = N_i(u) P_i.
     * 
     * @param points Evaluation points in parametric coordinates as a (Q × tdim) matrix.
     * @param order The highest order of parametric derivatives to compute.
     * @return A vector of matrices ordered lexicographically by derivative degree.
     *         The size of each matrix is (Q, gdim).
     * 
     *      Let tdim=2 (surface patch geometry) and let parametric coordinates be (u,v):
     * 
     *      n = 0 : (0,0) -> [X]       (Physical coordinates / Position)
     *      n = 1 : (1,0) -> [dX/du]
     *      n = 2 : (0,1) -> [dX/dv]
     *      n = 3 : (2,0) -> [d2X/du2]
     *      n = 4 : (1,1) -> [d2X/dudv]
     *      n = 5 : (0,2) -> [d2X/dv2]
     *      ...
     */
    virtual std::vector<Eigen::MatrixXd> eval_geometry(const Eigen::MatrixXd& points, 
                                                       std::size_t order = 0) const = 0;

    /**
     * @brief Evaluate the Jacobian determinant (integration measure) at each point.
     * * The Jacobian determinant represents the ratio of the physical "volume" 
     * (length, area, or volume) to the parametric "volume".
     * * - If tdim = 1: Returns the arc-length increment ||dX/du||.
     * - If tdim = 2: Returns the area increment ||dX/du × dX/dv||.
     * - If tdim = 3: Returns the volume increment |det(J)|.
     * * @param points Evaluation points in parametric coordinates as a (Q × tdim) matrix.
     * @return A vector of size (Q) containing the Jacobian determinant at each point.
     */
    virtual Eigen::VectorXd eval_jacobian(const Eigen::MatrixXd& points) const = 0;

protected:

    // === Member Variables ===========================================================

    /// @brief Geometric dimension of the patch (e.g., 3 for surfaces in 3D)
    std::size_t gdim_;

    /// @brief Topological dimension of the patch (e.g., 2 for surfaces)
    std::size_t tdim_;

    /// @brief Control points of the patch, stored as a matrix where each row is a control point
    Eigen::MatrixXd control_pts_;

};

} // namespace pyck

#endif // PYCK_PATCH_HPP