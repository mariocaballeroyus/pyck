#ifndef PYCK_MASTER_SLAVE_CONSTRAINT_HPP
#define PYCK_MASTER_SLAVE_CONSTRAINT_HPP

#include <vector>
#include <utility>

#include "constraint.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Enforces a 1-to-1 master-slave relationship between pairs of DOFs.
 *        For each pair (master, slave), strongly enforces `u_slave = u_master`
 *        while maintaining system symmetry.
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class MasterSlaveConstraint : public Constraint<T>
{
public:

    /**
     * @brief Construct a MasterSlaveConstraint from a list of (master, slave) DOF pairs.
     *
     * @param master_slave_pairs List of pairs where first is master DOF, second is slave DOF.
     */
    explicit MasterSlaveConstraint(std::vector<std::pair<Index, Index>> master_slave_pairs);

    /**
     * @brief Apply the master-slave constraints to the stiffness matrix and load vector
     *        in a way that preserves symmetry.
     *
     * @param stiffness Stiffness matrix
     * @param load Load vector
     */
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;

    /// @brief Get the master-slave pairs
    const std::vector<std::pair<Index, Index>>& pairs() const { return pairs_; }

private:
    std::vector<std::pair<Index, Index>> pairs_;
};

} // namespace pyck

#endif // PYCK_MASTER_SLAVE_CONSTRAINT_HPP
