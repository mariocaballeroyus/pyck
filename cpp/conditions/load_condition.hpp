#ifndef PYCK_LOAD_CONDITION_HPP
#define PYCK_LOAD_CONDITION_HPP

#include <functional>
#include <vector>
#include <Eigen/Dense>

#include "condition.hpp"
#include "patch.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Generic external load condition evaluating the integral: 
///        f_ext = \int_{\Gamma} N^T t d\Gamma
///        where N are the shape functions and t is the external force/traction.
template <std::floating_point T, std::size_t d>
class LoadCondition : public Condition<T>
{
public:
    /// @brief Construct a generic load condition by numerically integrating the load function.
    /// @param patch The geometric patch over which to integrate.
    /// @param quadrature The quadrature rule used for integration.
    /// @param load_fn The external scalar traction function t(u).
    LoadCondition(const Patch<T, d>& patch,
                  const QuadratureRule<T, d>& quadrature,
                  const std::function<T(const ColMatrix<T, d>&)>& load_fn);

    /// @brief Apply the integrated load to the global load vector.
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;

private:
    std::vector<Index> global_dofs_;
    Vector<T> element_load_;
};

} // namespace pyck

#endif // PYCK_LOAD_CONDITION_HPP