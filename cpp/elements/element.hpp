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

    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const = 0;

    /// @brief Transverse displacement shape matrix N_w (Q × n).
    ///
    /// Maps nodal transverse-displacement DOFs to interpolated values at the
    /// quadrature points.  Used in the load-vector integral ∫ N_w^T q dΩ.
    virtual Matrix<T> transverse_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Compute the generalized strain-displacement matrix B.
    /// @param shape_derivs Pre-evaluated shape functions and their derivatives.
    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Number of DOFs per node.
    virtual std::size_t num_node_dofs() const = 0;

    /// @brief Required derivative order for shape function evaluation.
    virtual std::size_t min_order() const = 0;

};

/// @brief Specialized base class for 1D elements.
template <std::floating_point T>
class Element<T, 1>
{
public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const = 0;

    /// @brief Transverse displacement shape matrix N_w (Q × n).
    ///
    /// Maps nodal transverse-displacement DOFs to interpolated values at the
    /// quadrature points.  Used in the load-vector integral ∫ N_w^T q dΩ.
    virtual Matrix<T> transverse_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Number of DOFs per node
    virtual std::size_t num_node_dofs() const = 0;

    /// @brief Minimum required derivative order for shape function evaluation
    virtual std::size_t min_order() const { return 0; }

protected:
    /// @brief Derivative mapping
    enum idx { val = 0, d1 = 1, d11 = 2, d111 = 3 };
};

/// @brief Specialized base class for 2D elements.
template <std::floating_point T>
class Element<T, 2>
{
public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const = 0;

    /// @brief Transverse displacement shape matrix N_w (Q × n).
    ///
    /// Maps nodal transverse-displacement DOFs to interpolated values at the
    /// quadrature points.  Used in the load-vector integral ∫ N_w^T q dΩ.
    virtual Matrix<T> transverse_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual std::size_t num_node_dofs() const = 0;

    virtual std::size_t min_order() const { return 0; }

protected:
    /// @brief Derivative mapping
    enum idx { val = 0, d1 = 1, d2 = 2, d11 = 3, d12 = 4, d22 = 5, d111 = 6, d112 = 7, d122 = 8, d222 = 9 };
};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
