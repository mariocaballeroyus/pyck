#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP


#include <vector>
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
                                         Matrix<T>& stiffness) const = 0;

    virtual void compute_local_load(const Patch<T, d>& patch,
                                    const ColMatrix<T, d>& q_points,
                                    const Vector<T>& q_weights,
                                    const Vector<T>& load_values,
                                    Vector<T>& load) const = 0;

};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
