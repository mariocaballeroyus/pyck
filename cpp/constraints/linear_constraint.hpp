#ifndef PYCK_LINEAR_CONSTRAINT_HPP
#define PYCK_LINEAR_CONSTRAINT_HPP

#include "constraint.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Linear multifreedom constraint:
 *        u[slaves[k]] = sum_j(weights[j] * u[masters(k, j)]) + constant.
 *
 * Condenses slave DOFs into master DOFs using in-place elimination,
 * maintaining system symmetry.
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class LinearConstraint : public Constraint<T>
{
public:

    /**
     * @brief Construct a LinearConstraint over a set of slave DOFs.
     *
     *        u_slave[k] = sum(weights[i] * u_masters(k, i)) + c
     *
     * Applies in-place elimination to condense slave DOFs into master DOFs,
     * maintaining system symmetry.
     *
     * @param slaves   Slave DOF indices.
     * @param masters  Matrix of master DOFs (rows = slaves, cols = n_masters).
     * @param weights  Shared coefficient weights.
     * @param constant Shared non-homogeneous offset.
     */
    LinearConstraint(IndexVector slaves,
                     IndexMatrix masters,
                     Vector<T>   weights,
                     T constant = 0.0);

    /**
     * @brief Apply the constraints to the stiffness matrix and load vector.
     */
    void apply(SparseMatrix<T>& stiffness, Vector<T>& load) const override;

    /// @brief Get the slave DOF indices
    const IndexVector& slaves() const { return slaves_; }

    /// @brief Get the master DOF matrix
    const IndexMatrix& masters() const { return masters_; }

    /// @brief Get the weights
    const Vector<T>& weights() const { return weights_; }

    /// @brief Get the constant
    T constant() const { return constant_; }

private:
    /// @brief Slave DOF indices
    IndexVector slaves_;

    /// @brief Master DOF indices (rows = slaves, cols = n_masters)
    IndexMatrix masters_;

    /// @brief Shared coefficient weights
    Vector<T> weights_;

    /// @brief Shared non-homogeneous offset
    T constant_;
};

} // namespace pyck

#endif // PYCK_LINEAR_CONSTRAINT_HPP
