#include "clamped.hpp"

namespace pyck
{

namespace conditions
{

template <std::floating_point T>
void apply_clamped_bc(const std::vector<Index>& dofs,
                      Matrix<T>& stiffness,
                      Vector<T>& load)
{
    // For a fully clamped B-spline:
    // Zero displacement and zero rotation at the boundary implies the first two 
    // (or last two) control points must be 0 exactly.

    for (Index dof : dofs)
    {
        // Zero out i-th row and column
        // K(i, :) = 0, K(:, i) = 0
        stiffness.row(dof).setZero();
        stiffness.col(dof).setZero();

        // Reduce i-th equation to trivial u_i = 0.0
        // K(i, i) = 1, f(i) = 0.0
        stiffness(dof, dof) = 1.0;
        load(dof) = 0.0;
    }
}

// === Template Instantiations ========================================================

template void apply_clamped_bc<double>(const std::vector<Index>&, Matrix<double>&, Vector<double>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template void apply_clamped_bc<float>(const std::vector<Index>&, Matrix<float>&, Vector<float>&);
#endif

} // namespace conditions

} // namespace pyck
