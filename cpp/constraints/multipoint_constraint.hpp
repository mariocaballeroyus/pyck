#ifndef PYCK_MULTIPOINT_CONSTRAINT_HPP
#define PYCK_MULTIPOINT_CONSTRAINT_HPP

#include <vector>
#include <utility>

#include "constraint.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Enforces a multipoint constraint: u_slave = sum(alpha_i * u_{master_i}) + c
 *        Statically condenses the slave out of the global system while maintaining
 *        symmetry of the stiffness matrix.
 *
 * @tparam T Scalar floating point type
 */
template <std::floating_point T>
class MultipointConstraint : public Constraint<T>
{
public:

    /**
     * @brief Defines one multi-point constraint equation.
     */
    struct Relation {
        Index slave;
        std::vector<std::pair<Index, T>> masters_weights;
        T constant;
    };

    /**
     * @brief Construct a MultipointConstraint with a list of relations.
     *
     * @param relations List of relations to enforce.
     */
    explicit MultipointConstraint(std::vector<Relation> relations);

    /**
     * @brief Construct a single MultipointConstraint.
     *
     * @param slave The slave degree of freedom index.
     * @param masters_weights List of (master_dof, alpha) pairs.
     * @param constant The constant term c.
     */
    MultipointConstraint(Index slave, std::vector<std::pair<Index, T>> masters_weights, T constant = 0.0);

    /**
     * @brief Apply the multi-point constraints to the stiffness matrix and load vector
     *        in a way that preserves symmetry.
     *
     * @param stiffness Stiffness matrix
     * @param load Load vector
     */
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;

    /// @brief Get the relations
    const std::vector<Relation>& relations() const { return relations_; }

private:
    std::vector<Relation> relations_;
};

} // namespace pyck

#endif // PYCK_MULTIPOINT_CONSTRAINT_HPP
