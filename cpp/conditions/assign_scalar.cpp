#include "assign_scalar.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
void assign_scalar(const std::vector<Index>& dofs,
                   const std::vector<T>& values,
                   Matrix<T>& stiffness,
                   Vector<T>& load)
{
    if (dofs.size() != values.size()) {
        throw std::invalid_argument("Size mismatch: number of dofs must equal number of prescribed values.");
    }

    // Move the influence from the LHS to the RHS load vector
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
        Index dof = dofs[i];
        T val = values[i];
        
        // Only adjust if the prescribed value is non-zero, avoiding floating math overhead.
        if (val != T(0.0)) {
            // load = load - K.col(dof) * value
            load.noalias() -= stiffness.col(dof) * val;
        }
    }

    // Zero out the row and column precisely, enforcing exact constraint satisfaction 
    // u_i = value without compromising global bounds or solver algorithms expecting symmetry.
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
        Index dof = dofs[i];
        T val = values[i];

        // Zero out the entire i-th row and column
        stiffness.row(dof).setZero();
        stiffness.col(dof).setZero();

        // Put down the trivial equation K_ii * u_i = f_i
        stiffness(dof, dof) = 1.0;
        load(dof) = val;
    }
}

template <std::floating_point T>
void assign_zeros(const std::vector<Index>& dofs,
                  Matrix<T>& stiffness,
                  Vector<T>& load)
{
    // The influence of the boundary is zero because the boundary conditions are zero. 
    // Thus, no shifting to RHS is required. We perform a faster single loop symmetry step.

    for (Index dof : dofs)
    {
        // Zero out the entire i-th row and column
        stiffness.row(dof).setZero();
        stiffness.col(dof).setZero();

        // Target: u_i = 0.0
        stiffness(dof, dof) = 1.0;
        load(dof) = 0.0;
    }
}

// === Template Instantiations ========================================================

template void assign_scalar<double>(const std::vector<Index>&, const std::vector<double>&, Matrix<double>&, Vector<double>&);
template void assign_zeros<double>(const std::vector<Index>&, Matrix<double>&, Vector<double>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template void assign_scalar<float>(const std::vector<Index>&, const std::vector<float>&, Matrix<float>&, Vector<float>&);
template void assign_zeros<float>(const std::vector<Index>&, Matrix<float>&, Vector<float>&);
#endif

} // namespace pyck
