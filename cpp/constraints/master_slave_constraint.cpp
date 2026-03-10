#include "master_slave_constraint.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
MasterSlaveConstraint<T>::MasterSlaveConstraint(std::vector<std::pair<Index, Index>> master_slave_pairs)
    : pairs_(std::move(master_slave_pairs))
{
}

template <std::floating_point T>
void MasterSlaveConstraint<T>::apply(Matrix<T>& K, Vector<T>& F) const
{
    // Apply constraints sequentially for each pair. 
    // This allows correctly enforcing multiple independent pairs.
    for (const auto& [m, s] : pairs_) {
        if (m == s) {
            continue; // Nothing to do if master == slave
        }
        if (m >= K.rows() || s >= K.rows()) {
            throw std::out_of_range("MasterSlaveConstraint: DOF index out of bounds.");
        }

        // Add row s to row m, and add F(s) to F(m)
        K.row(m) += K.row(s);
        F(m) += F(s);

        // Add col s to col m
        K.col(m) += K.col(s);

        // Zero out row s and col s
        K.row(s).setZero();
        K.col(s).setZero();

        // Set K(s, s) = 1, K(s, m) = -1, K(m, s) = -1
        K(s, s) = 1.0;
        K(s, m) = -1.0;
        K(m, s) = -1.0;

        // Add 1 to K(m, m) to balance the equation and maintain symmetry
        K(m, m) += 1.0;

        // Set F(s) = 0
        F(s) = 0.0;
    }
}

// === Template Instantiations ========================================================

template class MasterSlaveConstraint<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class MasterSlaveConstraint<float>;
#endif

} // namespace pyck
