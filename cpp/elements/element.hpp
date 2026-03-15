#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP


#include <vector>
#include <array>
#include <Eigen/Dense>

#include "patch.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Abstract base class for elements.
/// @tparam T Scalar type.
/// @tparam d Dimension.
template <std::floating_point T, std::size_t d>
class Element
{

public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const Patch<T, d>& patch,
                                         const ColMatrix<T, d>& q_points,
                                         const Vector<T>& q_weights,
                                         Index span,
                                         Matrix<T>& stiffness) const = 0;

    /// @brief Compute the generalized shape function matrix N.
    /// @param shape_derivs Pre-evaluated shape functions and their derivatives.
    virtual Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Compute the generalized strain-displacement matrix B.
    /// @param shape_derivs Pre-evaluated shape functions and their derivatives.
    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Get the number of degrees of freedom per node.
    virtual std::size_t num_dofs_per_node() const = 0;

    /// @brief Get the minimum required derivative order for shape_matrix evaluation in LoadCondition.
    virtual std::size_t required_shape_order() const { return 0; }

};

/// @brief Specialized base class for 1D elements.
template <std::floating_point T>
class Element<T, 1>
{
public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const Patch<T, 1>& patch,
                                         const ColMatrix<T, 1>& q_points,
                                         const Vector<T>& q_weights,
                                         Index span,
                                         Matrix<T>& stiffness) const = 0;

    virtual Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;
    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;
    virtual std::size_t num_dofs_per_node() const = 0;
    virtual std::size_t required_shape_order() const { return 0; }

protected:
    /// @brief Derivative mapping
    enum idx { fn = 0, u = 1, uu = 2, uuu = 3 };
};

/// @brief Specialized base class for 2D elements.
template <std::floating_point T>
class Element<T, 2>
{
public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const Patch<T, 2>& patch,
                                         const ColMatrix<T, 2>& q_points,
                                         const Vector<T>& q_weights,
                                         Index span,
                                         Matrix<T>& stiffness) const = 0;

    virtual Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;
    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;
    virtual std::size_t num_dofs_per_node() const = 0;
    virtual std::size_t required_shape_order() const { return 0; }

protected:
    /// @brief Derivative mapping
    enum idx { fn = 0, u = 1, v = 2, uu = 3, uv = 4, vv = 5, uuu = 6, uuv = 7, uvv = 8, vvv = 9 };
};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
