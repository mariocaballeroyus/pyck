#ifndef PYCK_PATCH_HPP
#define PYCK_PATCH_HPP

#include <utility>
#include <vector>

#include "basis.hpp"

namespace pyck
{

/**
 * Abstract base class for parametric surface patches defined by tensor the 
 * tensor product of one-dimensional basis functions.
 */
class Patch
{
public:
    /// @brief Virtual destructor
    virtual ~Patch() = default;

    Patch(std::size_t gdim, 
          std::size_t tdim, 
          const Eigen::MatrixXd& control_points)
        : gdim_(gdim), tdim_(tdim), control_points_(control_points) {}

    /// @brief Get the geometric dimension of the patch
    std::size_t gdim() const { return gdim_; }

    /// @brief Get the topological dimension of the patch
    std::size_t tdim() const { return tdim_; }

    /// @brief Get the control points of the patch
    const Eigen::MatrixXd& control_points() const { return control_points_; }

    /// @brief Get the number of control points in the patch
    std::size_t num_control_points() const { return control_points_.size(); }

    /// @brief Get the control points of the patch (non-const version)
    Eigen::MatrixXd& control_points() { return control_points_; }

    /// @brief Get the basis functions for a given parametric direction
    /// @param dir The parametric direction
    virtual const Basis& basis(std::size_t dir) const = 0;

    /// @brief Compute the Jacobian matrices and their determinants at a
    ///        tensor-product grid of parameter values.
    /// @return A pair: vector of Jacobian matrices (gdim × tdim) and
    ///         a VectorXd of sqrt(det(J^T J)) values.
    virtual std::pair<std::vector<Eigen::MatrixXd>, Eigen::VectorXd>
    jacobian(const Eigen::VectorXd& u, const Eigen::VectorXd& v) const = 0;

protected:
    /// @brief Geometric dimension of the patch (e.g., 3 for surfaces in 3D)
    std::size_t gdim_;

    /// @brief Topological dimension of the patch (e.g., 2 for surfaces)
    std::size_t tdim_;

    /// @brief Control points of the patch, stored as a matrix where each row is a control point
    Eigen::MatrixXd control_points_;
};

} // namespace pyck

#endif // PYCK_PATCH_HPP