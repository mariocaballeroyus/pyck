#ifndef PYCK_DIRECTORS_HPP
#define PYCK_DIRECTORS_HPP

#include <utility>
#include <concepts>

#include "../types.hpp"
#include "local_frame.hpp"

namespace pyck
{

/**
 * @brief Director vectors for a parametric surface via cross product of the
 *        tangent vectors.
 */
template <std::floating_point T>
ColMatrix<T, 3>
eval_normal(const LocalFrame<T, 2>& local);

/**
 * @brief Director derivatives (a_{3,1}, a_{3,2}) via Weingarten:
 *        a_{3,β} = -g^{γδ} (a_{δβ}·a_3) a_γ.
 */
template <std::floating_point T>
std::pair<ColMatrix<T, 3>, ColMatrix<T, 3>>
eval_normal_deriv(const LocalFrame<T, 2>& local,
                  const ColMatrix<T, 3>& a_3);

} // namespace pyck

#endif // PYCK_DIRECTORS_HPP
