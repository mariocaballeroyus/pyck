#ifndef PYCK_GEOMETRY_EVALUATION_HPP
#define PYCK_GEOMETRY_EVALUATION_HPP

#include <vector>
#include <Eigen/Dense>
#include "patch.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Evaluate shape functions at given parametric points (automatic span finding).
 * @param patch The patch to evaluate.
 * @param points Parametric coordinates.
 * @param order Derivative order.
 * @return Vector of matrices containing shape function values and derivatives.
 */
template <std::floating_point T, std::size_t d>
std::vector<Matrix<T>> eval_shape_at(const Patch<T, d>& patch, 
                                     const ColMatrix<T, d>& points, 
                                     std::size_t order = 0);

/**
 * @brief Evaluate physical coordinates at given parametric points (automatic span finding).
 * @param patch The patch to evaluate.
 * @param points Parametric coordinates.
 * @return Physical coordinates in 3D space.
 */
template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> eval_geometry_at(const Patch<T, d>& patch, 
                                 const ColMatrix<T, d>& points);

} // namespace pyck

#endif // PYCK_GEOMETRY_EVALUATION_HPP
