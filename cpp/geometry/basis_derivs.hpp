#ifndef PYCK_BASIS_DERIVS_HPP
#define PYCK_BASIS_DERIVS_HPP

#include <cstddef>
#include <concepts>

#include "../types.hpp"
#include "patch.hpp"

namespace pyck
{

/**
 * @brief Parametric basis derivatives at a set of quadrature points
 *        defined on a single span of a patch.
 */
template <std::floating_point T, std::size_t d>
struct BasisDerivs;

/**
 * @brief Parametric basis derivatives defined on 1-dimensional patches.
 */
template <std::floating_point T>
struct BasisDerivs<T, 1>
{
    Matrix<T> N;          ///< values                                             (Q×K)
    Matrix<T> N_u;        ///< 1st derivative N_{,u}             (order ≥ 1)      (Q×K)
    Matrix<T> N_uu;       ///< 2nd derivative N_{,uu}            (order ≥ 2)      (Q×K)
    Matrix<T> N_uuu;      ///< 3rd derivative N_{,uuu}           (order ≥ 3)      (Q×K)
    Index     order = 0;
};

/**
 * @brief Basis derivatives defined on 2-dimensional patches.
 */
template <std::floating_point T>
struct BasisDerivs<T, 2>
{
    Matrix<T> N;                           ///< values                            (Q×K)
    Matrix<T> N_u,   N_v;                  ///< 1st derivatives  (order ≥ 1)      (Q×K)
    Matrix<T> N_uu,  N_uv,  N_vv;          ///< 2nd derivatives  (order ≥ 2)      (Q×K)
    Matrix<T> N_uuu, N_uuv, N_uvv, N_vvv;  ///< 3rd derivatives  (order ≥ 3)      (Q×K)
    Index     order = 0;
};

/**
 * @brief Evaluate shape functions and their derivatives at the given coordinates
 *        in the parametric domain.
 *
 * @param patch        The patch carrying the tensor-product basis.
 * @param eval_coords  Coordinates in the parametric domain.
 * @param span_idx     Index of the span to evaluate.
 * @param derivs_order Highest order of derivatives to compute.
 * @return A struct containing the basis functions and their derivatives.
 */
template <std::floating_point T>
BasisDerivs<T, 1>
eval_basis(const Patch<T, 1>& patch,
           const ColMatrix<T, 1>& eval_coords,
           Index span_idx,
           std::size_t derivs_order = 0);

template <std::floating_point T>
BasisDerivs<T, 2>
eval_basis(const Patch<T, 2>& patch,
           const ColMatrix<T, 2>& eval_coords,
           Index span_idx,
           std::size_t derivs_order = 0);

} // namespace pyck

#endif // PYCK_BASIS_DERIVS_HPP
