#ifndef PYCK_CONSTRAINT_HPP
#define PYCK_CONSTRAINT_HPP

#include "../types.hpp"

namespace pyck
{

template <std::floating_point T>
class Constraint
{
public:

    /**
     * @brief Virtual destructor
     */
    virtual ~Constraint() = default;

    /**
     * @brief Apply the constraint to the stiffness matrix and load vector
     *
     * @param stiffness Stiffness matrix
     * @param load Load vector
     */
    virtual void apply(Matrix<T>& stiffness, Vector<T>& load) const = 0;

};

} // namespace pyck

#endif // PYCK_CONSTRAINT_HPP
