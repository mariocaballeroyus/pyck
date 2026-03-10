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

};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
