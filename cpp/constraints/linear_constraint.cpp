#include "linear_constraint.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T>
LinearConstraint<T>::LinearConstraint(IndexVector slaves,
                                      IndexMatrix masters,
                                      Vector<T>   weights,
                                      T constant)
    : slaves_(std::move(slaves)),
      masters_(std::move(masters)),
      weights_(std::move(weights)),
      constant_(constant)
{
    const Index n_slaves = slaves_.size();
    if (masters_.rows() != static_cast<Index>(n_slaves)) {
        throw std::invalid_argument("LinearConstraint: masters rows "
                                    "must match slaves size.");
    }

    const Index n_masters = weights_.size();
    if (n_masters == 0 && n_slaves > 0) {
        throw std::invalid_argument("LinearConstraint: weights cannot be empty "
                                    "if slaves are provided.");
    }

    if (masters_.cols() != static_cast<Index>(n_masters)) {
        throw std::invalid_argument("LinearConstraint: masters columns "
                                    "must match weights size.");
    }

    for (Index k = 0; k < n_slaves; ++k) {
        const Index s = slaves_(k);
        for (Index i = 0; i < n_masters; ++i) {
            if (s == masters_(k, i)) {
                throw std::invalid_argument("LinearConstraint: "
                                            "slave cannot be its own master.");
            }
        }
    }
}

template <std::floating_point T>
void LinearConstraint<T>::apply(Matrix<T>& K, Vector<T>& F) const
{
    const Index n_slaves  = slaves_.size();
    const Index n_masters = weights_.size();

    // Pull out constant load update
    if (constant_ != 0.0) {
        for (Index k = 0; k < n_slaves; ++k) {
            const Index s = slaves_(k);
            // F_j -= K_js * c
            F.noalias() -= K.col(s) * constant_;
        }
    }

    for (Index k = 0; k < n_slaves; ++k) {
        const Index s = slaves_(k);
        const auto masters = masters_.row(k);

        // Distribute slave load to masters
        for (Index i = 0; i < n_masters; ++i) {
            const Index m = masters(i);
            F(m) += weights_(i) * F(s);
        }

        // Symmetric condensation of stiffness matrix
        for (Index i = 0; i < n_masters; ++i) {
            const Index m = masters(i);
            K.row(m) += weights_(i) * K.row(s);
            K.col(m) += weights_(i) * K.col(s);
        }

        // Zero out slave row and column
        K.row(s).setZero();
        K.col(s).setZero();

        // Enforce constraint equation: u_s - sum(w_m * u_m) = c
        for (Index i = 0; i < n_masters; ++i) {
            const Index m = masters(i);
            const T weight = weights_(i);
            K(s, m) = -weight;
            K(m, s) = -weight;
        }
        K(s, s) = 1.0;
        F(s) = constant_;
    }
}

// === Template Instantiations ========================================================

template class LinearConstraint<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LinearConstraint<float>;
#endif

} // namespace pyck
