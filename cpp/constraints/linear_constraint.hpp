#ifndef PYCK_LINEAR_CONSTRAINT_HPP
#define PYCK_LINEAR_CONSTRAINT_HPP

#include <vector>
#include <utility>

#include "constraint.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Linear multifreedom constraint: u[slaves[k]] = sum_j(weights[j] * u[masters(k, j)]) + constant.
 *
 * Condenses slave DOFs into master DOFs using in-place elimination, maintaining 
 * system symmetry.
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class LinearConstraint : public Constraint<T>
{
public:

    /**
     * @brief Construct a LinearConstraint u_slave[k] = sum(weights[i] * u_masters(k, i)) + c 
     *        over a set of slave DOFs.
     *
     *        Applies in-place elimination to condense slave DOFs into master DOFs, 
     *        maintaining system symmetry.
     *
     * @param slaves List of slave DOF indices.
     * @param masters Matrix of master DOFs (rows = slaves, cols = n_masters).
     * @param weights Shared coefficient weights.
     * @param constant Shared non-homogeneous offset.
     */
    LinearConstraint(std::vector<Index> slaves,
                     IndexMatrix masters,
                     std::vector<T> weights,
                     T constant = 0.0);


    /**
     * @brief Apply the constraints to the stiffness matrix and load vector.
     */
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;

    /// @brief Get the constrained DOF indices
    const std::vector<Index>& slaves() const 
    { return slaves_; }

    /// @brief Get the master DOF indices
    const IndexMatrix& masters() const 
    { return masters_; }

    /// @brief Get the weights
    const std::vector<T>& weights() const 
    { return weights_; }

    /// @brief Get the constant
    T constant() const 
    { return constant_; }

private:
    /// @brief Slave DOF indices
    std::vector<Index> slaves_;

    /// @brief Master DOF indices
    IndexMatrix masters_;

    /// @brief Shared coefficient weights
    std::vector<T> weights_;

    /// @brief Shared non-homogeneous offset
    T constant_;
};

} // namespace pyck

#endif // PYCK_LINEAR_CONSTRAINT_HPP
