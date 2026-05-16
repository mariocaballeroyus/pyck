#ifndef PYCK_CHRISTOFFELS_HPP
#define PYCK_CHRISTOFFELS_HPP

#include <cstddef>
#include <concepts>

#include "../types.hpp"
#include "local_frame.hpp"

namespace pyck
{

/**
 * @brief Christoffel symbols Γ^γ_{αβ} at each quadrature point on a single
 *        span of a patch.
 */
template <std::floating_point T, std::size_t d>
struct ChristoffelSymbols;

/**
 * @brief 1D Christoffel symbol Γ^1_{11} = (a_1 · a_{1,1}) / g_{11}.
 */
template <std::floating_point T>
struct ChristoffelSymbols<T, 1>
{
    Vector<T> G1_11;                 ///< Γ^1_{11} (Q)
};

/**
 * @brief 2D Christoffel symbols Γ^γ_{αβ} = a^γ · a_{αβ}, symmetric in αβ.
 *        Six independent components per quadrature point.
 */
template <std::floating_point T>
struct ChristoffelSymbols<T, 2>
{
    Vector<T> G1_11, G1_12, G1_22;   ///< Γ^1_{αβ} (Q)
    Vector<T> G2_11, G2_12, G2_22;   ///< Γ^2_{αβ} (Q)
};

/**
 * @brief 3D Christoffel symbols Γ^γ_{αβ} = a^γ · a_{αβ}, symmetric in αβ.
 *        Eighteen independent components per quadrature point
 *        (γ ∈ {1,2,3}; (α,β) ∈ {(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)}).
 */
template <std::floating_point T>
struct ChristoffelSymbols<T, 3>
{
    Vector<T> G1_11, G1_12, G1_13, G1_22, G1_23, G1_33;   ///< Γ^1_{αβ} (Q)
    Vector<T> G2_11, G2_12, G2_13, G2_22, G2_23, G2_33;   ///< Γ^2_{αβ} (Q)
    Vector<T> G3_11, G3_12, G3_13, G3_22, G3_23, G3_33;   ///< Γ^3_{αβ} (Q)
};

/**
 * @brief Christoffel symbols Γ^γ_{αβ} at each quadrature point.
 *
 * 1D returns the single non-zero symbol Γ¹_{11}; 2D returns all six
 * independent components Γ^γ_{αβ} (symmetric in αβ); 3D returns all
 * eighteen independent components.
 */
template <std::floating_point T>
ChristoffelSymbols<T, 1>
eval_christoffel(const LocalFrame<T, 1>& local);

template <std::floating_point T>
ChristoffelSymbols<T, 2>
eval_christoffel(const LocalFrame<T, 2>& local);

template <std::floating_point T>
ChristoffelSymbols<T, 3>
eval_christoffel(const LocalFrame<T, 3>& local);

} // namespace pyck

#endif // PYCK_CHRISTOFFELS_HPP
