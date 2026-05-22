#ifndef PYCK_TYPES_HPP
#define PYCK_TYPES_HPP

#include <cstddef>
#include <concepts>
#include <memory>
#include <Eigen/Dense>

namespace pyck
{

/// @brief Signed index type
using Index = int;

/// @brief Row-major dynamic-sized dense matrix of type T
template <std::floating_point T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/// @brief Shared pointer type
template <typename T>
using Ptr = std::shared_ptr<T>;

/// @brief Dynamic-sized column vector of type T
template <std::floating_point T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

/// @brief Dynamic-sized row vector of type T
template <std::floating_point T>
using RowVector = Eigen::RowVector<T, Eigen::Dynamic>;

/// @brief Dynamic-sized dense matrix of integers (typically for DOFs)
using IndexMatrix = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// @brief Dynamic-sized column vector of integers (typically for DOF indices)
using IndexVector = Eigen::Vector<Index, Eigen::Dynamic>;

/// @brief Dense matrix of type T with dynamic number of rows and fixed number of columns
template <std::floating_point T, std::size_t Cols>
using ColMatrix = Eigen::Matrix<T, Eigen::Dynamic, Cols>;

/// @brief Dense matrix of type T with fixed number of rows and dynamic number of columns
template <std::floating_point T, std::size_t Rows>
using RowMatrix = Eigen::Matrix<T, Rows, Eigen::Dynamic>;

/// @brief Vector of type T with a fixed compile-time size N
template <std::floating_point T, std::size_t N>
using StaticVector = Eigen::Vector<T, N>;

/// @brief Maximum strain dimension across all Element<T,2> overrides.
///        Used as the storage bound for ConstitutiveMatrix.
///        Bump when introducing an element whose strain dim exceeds this.
constexpr int MAX_STRAIN_DIM = 8;

/// @brief Stack-resident constitutive (D) matrix returned by
///        Element::constitutive_matrix. Dynamic runtime dimension, with
///        storage sized for MAX_STRAIN_DIM × MAX_STRAIN_DIM (the worst
///        case across all element formulations) held inline in the matrix
///        object — so the type is uniform under virtual dispatch yet
///        allocation-free.
template <std::floating_point T>
using ConstitutiveMatrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, MAX_STRAIN_DIM, MAX_STRAIN_DIM>;

/// @brief Dynamic-sized dense array of type T
template <std::floating_point T>
using Array = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

/// @brief Dense array of type T with dynamic number of rows and fixed number of columns
template <std::floating_point T, std::size_t Cols>
using ColArray = Eigen::Array<T, Eigen::Dynamic, Cols>;

/// @brief Dense array of type T with fixed number of rows and dynamic number of columns
template <std::floating_point T, std::size_t Rows>
using RowArray = Eigen::Array<T, Rows, Eigen::Dynamic>;

} // namespace pyck

#endif // PYCK_TYPES_HPP
