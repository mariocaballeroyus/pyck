#include "strong_dirichlet_constraint.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
StrongDirichletConstraint<T>::StrongDirichletConstraint(std::vector<Index> dofs, std::vector<T> values)
    : dofs_(std::move(dofs)), values_(std::move(values))
{
    if (dofs_.size() != values_.size()) {
        throw std::invalid_argument("StrongDirichletConstraint: "
                                    "dofs and values must have the same size.");
    }
}

template <std::floating_point T>
StrongDirichletConstraint<T>::StrongDirichletConstraint(std::vector<Index> dofs, T value)
    : dofs_(dofs), values_(dofs.size(), value) {}

template <std::floating_point T>
void StrongDirichletConstraint<T>::apply(Matrix<T>& stiffness, Vector<T>& load) const
{
    // Move the influence from the LHS to the RHS load vector
    for (std::size_t i = 0; i < dofs_.size(); ++i)
    {
        Index dof = dofs_[i];
        T val = values_[i];
        
        // Only adjust if the prescribed value is non-zero, avoiding floating math overhead.
        if (val != T(0.0)) {
            // load = load - K.col(dof) * value
            load.noalias() -= stiffness.col(dof) * val;
        }
    }

    // Zero out the row and column precisely, enforcing exact constraint satisfaction 
    // u_i = value without compromising global bounds or solver algorithms expecting symmetry.
    for (std::size_t i = 0; i < dofs_.size(); ++i)
    {
        Index dof = dofs_[i];
        T val = values_[i];

        // Zero out the entire i-th row and column
        stiffness.row(dof).setZero();
        stiffness.col(dof).setZero();

        // Put down the trivial equation K_ii * u_i = f_i
        stiffness(dof, dof) = 1.0;
        load(dof) = val;
    }
}

// === Template Instantiations ========================================================

template class StrongDirichletConstraint<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class StrongDirichletConstraint<float>;
#endif

} // namespace pyck
