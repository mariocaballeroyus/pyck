#include "direct_constraint.hpp"

#include <vector>

namespace pyck
{

template <std::floating_point T>
DirectConstraint<T>::DirectConstraint(IndexVector dofs,
                                      Vector<T>   values)
    : dofs_(std::move(dofs)),
      values_(std::move(values))
{
    if (dofs_.size() != values_.size()) {
        throw std::invalid_argument("DirectConstraint: "
                                    "dofs and values must have the same size.");
    }
}

template <std::floating_point T>
DirectConstraint<T>::DirectConstraint(IndexVector dofs,
                                      T value)
    : dofs_(std::move(dofs)),
      values_(Vector<T>::Constant(dofs_.size(), value)) {}

template <std::floating_point T>
void DirectConstraint<T>::apply(SparseMatrix<T>& stiffness,
                                Vector<T>& load) const
{
    const Index n    = dofs_.size();
    const Index ndof = static_cast<Index>(stiffness.rows());

    // Move the influence of the prescribed values from the LHS to the RHS
    Vector<T> g = Vector<T>::Zero(ndof);
    for (Index i = 0; i < n; ++i) {
        g(dofs_(i)) = values_(i);
    }
    load.noalias() -= stiffness * g;

    // Mark constrained DOFs.
    std::vector<bool> fixed(static_cast<std::size_t>(ndof), false);
    for (Index i = 0; i < n; ++i) {
        fixed[static_cast<std::size_t>(dofs_(i))] = true;
    }

    // Symmetric elimination: zero every entry sharing a constrained row or
    // column. Only values are touched, so iterators stay valid.
    for (Index k = 0; k < stiffness.outerSize(); ++k) {
        for (typename SparseMatrix<T>::InnerIterator it(stiffness, k); it; ++it) {
            if (fixed[static_cast<std::size_t>(it.row())] ||
                fixed[static_cast<std::size_t>(it.col())]) {
                it.valueRef() = T(0);
            }
        }
    }

    // Enforce the trivial equation u_i = value for each constrained DOF.
    for (Index i = 0; i < n; ++i) {
        const Index dof = dofs_(i);
        stiffness.coeffRef(dof, dof) = T(1);
        load(dof) = values_(i);
    }

    // Drop the explicit zeros left by the elimination.
    stiffness.prune(T(0));
}

// === Template Instantiations ========================================================

template class DirectConstraint<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class DirectConstraint<float>;
#endif

} // namespace pyck
