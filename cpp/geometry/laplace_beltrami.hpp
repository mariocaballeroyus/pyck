#ifndef PYCK_LAPLACE_BELTRAMI_HPP
#define PYCK_LAPLACE_BELTRAMI_HPP

#include <concepts>

#include "../types.hpp"
#include "local_frame.hpp"
#include "christoffels.hpp"

namespace pyck
{

/**
 * @brief Per-quadrature-point auxiliary data for ∂_α(Δ_g f) of a 2D scalar f.
 */
template <std::floating_point T>
struct LaplaceGradAux
{
    Vector<T> G11_d1, G12_d1, G22_d1;   ///< (g^{βγ})_{,1}                        (Q)
    Vector<T> G11_d2, G12_d2, G22_d2;   ///< (g^{βγ})_{,2}                        (Q)
    Vector<T> c1, c2;                   ///< c^δ ≡ g^{βγ} Γ^δ_{βγ}                (Q)
    Vector<T> c1_d1, c2_d1;             ///< (c^δ)_{,1}                          (Q)
    Vector<T> c1_d2, c2_d2;             ///< (c^δ)_{,2}                          (Q)
};

/**
 * @brief Build a LaplaceGradAux<T> at all quadrature points from a precomputed
 *        2D LocalFrame and ChristoffelSymbols. Requires `local.a111…a222` (the
 *        basis must have been evaluated at order ≥ 3).
 */
template <std::floating_point T>
LaplaceGradAux<T>
compute_laplace_grad_aux(const LocalFrame<T, 2>& local,
                         const ChristoffelSymbols<T, 2>& chr);

} // namespace pyck

#endif // PYCK_LAPLACE_BELTRAMI_HPP
