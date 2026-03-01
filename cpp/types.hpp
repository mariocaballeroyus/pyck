#ifndef PYCK_TYPES_HPP
#define PYCK_TYPES_HPP

#include <cstddef>
#include <concepts>
#include <Eigen/Dense>

namespace pyck
{

/// @brief Global integral type for indexing and sizes
using Index = std::size_t;

/// @brief Dynamic-sized column vector of type T
template <std::floating_point T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

/// @brief Dynamic-sized dense matrix of type T
template <std::floating_point T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/// @brief Dense matrix of type T with dynamic number of rows and fixed number of columns
template <std::floating_point T, std::size_t Cols>
using ColMatrix = Eigen::Matrix<T, Eigen::Dynamic, Cols>;

/// @brief Vector of type T with a fixed compile-time size N
template <std::floating_point T, std::size_t N>
using StaticVector = Eigen::Vector<T, N>;

} // namespace pyck

#endif // PYCK_TYPES_HPP
