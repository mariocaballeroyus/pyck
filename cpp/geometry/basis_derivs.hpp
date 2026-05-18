#ifndef PYCK_BASIS_DERIVS_HPP
#define PYCK_BASIS_DERIVS_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "../types.hpp"
#include "patch.hpp"

namespace pyck
{

/**
 * @brief Parametric basis derivatives at a set of quadrature points
 *        defined on a single span of a patch.
 *
 * Stored in nested `std::array` form indexed by parametric directions.
 * Symmetric components (`N_d2`, `N_d3`) populate only their upper-triangular
 * slots (i ≤ j, i ≤ j ≤ k); the remaining slots are default-initialised and
 * unused.
 */
template <std::floating_point T, std::size_t d>
struct BasisDerivs
{
    /// Shape function values N                                                   (Q×K).
    Matrix<T> N;

    /// First derivatives ∂_i N                                              (order ≥ 1).
    std::array<Matrix<T>, d> N_d1;

    /// Second derivatives ∂_{ij} N, sym; i ≤ j filled.           (order ≥ 2).
    std::array<std::array<Matrix<T>, d>, d> N_d2;

    /// Third derivatives ∂_{ijk} N, sym; i ≤ j ≤ k filled.           (order ≥ 3).
    std::array<std::array<std::array<Matrix<T>, d>, d>, d> N_d3;

    Index order = 0;
};

/**
 * @brief Evaluate shape functions and their derivatives at the given coordinates
 *        in the parametric domain.
 *
 * @param patch        The patch carrying the tensor-product basis.
 * @param eval_coords  Coordinates in the parametric domain, shape (Q, d).
 * @param span_idx     Flat index of the span to evaluate.
 * @param derivs_order Highest order of derivatives to compute.
 * @return A struct containing the basis functions and their derivatives.
 */
template <std::floating_point T, std::size_t d>
BasisDerivs<T, d>
eval_basis(const Patch<T, d>& patch,
           const std::type_identity_t<ColMatrix<T, d>>& eval_coords,
           Index span_idx,
           std::size_t derivs_order = 0);

} // namespace pyck

#endif // PYCK_BASIS_DERIVS_HPP
