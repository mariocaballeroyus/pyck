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
     * @brief Apply the constraint to the assembled stiffness matrix and load
     *        vector. Constraints run after the sparse matrix is built, since
     *        they perform row/column elimination on existing entries.
     *
     * @param stiffness Assembled global stiffness matrix
     * @param load Load vector
     */
    virtual void apply(SparseMatrix<T>& stiffness, Vector<T>& load) const = 0;

};

} // namespace pyck

#endif // PYCK_CONSTRAINT_HPP
