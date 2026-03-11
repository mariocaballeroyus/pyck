#include "multipoint_constraint.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
MultipointConstraint<T>::MultipointConstraint(std::vector<Relation> relations)
    : relations_(std::move(relations))
{
}

template <std::floating_point T>
MultipointConstraint<T>::MultipointConstraint(Index slave, std::vector<std::pair<Index, T>> masters_weights, T constant)
{
    relations_.push_back({slave, std::move(masters_weights), constant});
}

template <std::floating_point T>
void MultipointConstraint<T>::apply(Matrix<T>& K, Vector<T>& F) const
{
    // Apply constraints sequentially
    for (const auto& rel : relations_) {
        Index s = rel.slave;
        if (s >= K.rows()) {
            throw std::out_of_range("MultipointConstraint: slave DOF index out of bounds.");
        }
        for (const auto& [m, alpha] : rel.masters_weights) {
            if (m >= K.rows()) {
                throw std::out_of_range("MultipointConstraint: master DOF index out of bounds.");
            }
            if (s == m) {
                throw std::invalid_argument("MultipointConstraint: slave cannot be its own master.");
            }
        }

        Vector<T> K_s_row = K.row(s);
        Vector<T> K_s_col = K.col(s);
        T K_ss = K(s, s);
        T c = rel.constant;

        // F updates from constant c
        if (c != 0.0) {
            for (Index j = 0; j < K.rows(); ++j) {
                F(j) -= K_s_col(j) * c;
            }
        }

        // F updates for masters from slave RHS
        for (const auto& [m_i, alpha_i] : rel.masters_weights) {
            F(m_i) += alpha_i * F(s); // F(s) already includes the -K_ss * c update
        }

        // K updates: distribute slave row/col to masters
        for (const auto& [m_i, alpha_i] : rel.masters_weights) {
            K.row(m_i) += alpha_i * K_s_row;
        }
        for (const auto& [m_i, alpha_i] : rel.masters_weights) {
            K.col(m_i) += alpha_i * K_s_col;
        }
        
        // Exact matching for cross terms K(mi, mj) and artificial symmetry terms
        for (const auto& [m_i, alpha_i] : rel.masters_weights) {
            for (const auto& [m_j, alpha_j] : rel.masters_weights) {
                K(m_i, m_j) += alpha_i * alpha_j * K_ss;
                // Artificial symmetry for the new equation added to row s
                K(m_i, m_j) += alpha_i * alpha_j;
            }
            F(m_i) -= alpha_i * c;
        }

        // Zero out slave row and column
        K.row(s).setZero();
        K.col(s).setZero();

        // Enforce the constraint equation itself on the modified row/col 
        // u_s - sum(alpha_i * u_m_i) = c  =>  1 * u_s - alpha_i * u_m_i = c
        K(s, s) = 1.0;
        F(s) = c;

        for (const auto& [m_i, alpha_i] : rel.masters_weights) {
            K(s, m_i) = -alpha_i;
            K(m_i, s) = -alpha_i; // Symmetry
        }
    }
}

// === Template Instantiations ========================================================

template class MultipointConstraint<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class MultipointConstraint<float>;
#endif

} // namespace pyck
