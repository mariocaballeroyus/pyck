#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <functional>
#include <vector>
#include <Eigen/Dense>

#include "patch.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Abstract base class for elements.
/// @tparam T Scalar type.
/// @tparam d Dimension.
template <std::floating_point T, std::size_t d>
class Element
{

    using ScalarType = T;

public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const Patch<T, d>& patch,
                                         const QuadratureRule<T, d>& quadrature,
                                         Matrix<T>& stiffness) const = 0;

    virtual void compute_local_load(const Patch<T, d>& patch,
                                    const QuadratureRule<T, d>& quadrature,
                                    const std::function<ScalarType(const ColMatrix<T, d>&)>& load_fn,
                                    Vector<T>& load) const = 0;

};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
