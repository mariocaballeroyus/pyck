#include "direct_constraint.hpp"

namespace pyck
{

template <std::floating_point T>
DirectConstraint<T>::DirectConstraint(std::vector<Index> dofs, 
                                            std::vector<T> values)
    : dofs_(std::move(dofs)), 
      values_(std::move(values))
{
    if (dofs_.size() != values_.size()) {
        throw std::invalid_argument("DirectConstraint: "
                                    "dofs and values must have the same size.");
    }
}

template <std::floating_point T>
DirectConstraint<T>::DirectConstraint(std::vector<Index> dofs, 
                                            T value)
    : dofs_(std::move(dofs)),
      values_(dofs_.size(), value) {}

template <std::floating_point T>
void DirectConstraint<T>::apply(Matrix<T>& stiffness, 
                                   Vector<T>& load) const
{
    for (std::size_t i = 0; i < dofs_.size(); ++i)
    {
        Index dof = dofs_[i];
        T value = values_[i];
        
        if (value != T(0.0)) {
            // Move the influence from the LHS to the RHS load vector
            // f_i = f_i - K_ij * u_j
            load.noalias() -= stiffness.col(dof) * value;
        }
    }

    for (std::size_t i = 0; i < dofs_.size(); ++i)
    {
        Index dof = dofs_[i];
        T value = values_[i];

        // Zero out the row and column for the constrained DOF
        stiffness.row(dof).setZero();
        stiffness.col(dof).setZero();

        // Enforce trivial equation u_i = value
        stiffness(dof, dof) = 1.0;
        load(dof) = value;
    }
}

// === Template Instantiations ========================================================

template class DirectConstraint<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class DirectConstraint<float>;
#endif

} // namespace pyck
