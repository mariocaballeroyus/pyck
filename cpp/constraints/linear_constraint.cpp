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
void LinearConstraint<T>::apply(SparseMatrix<T>& K, Vector<T>& F) const
{
    const Index n         = static_cast<Index>(K.rows());
    const Index n_slaves  = slaves_.size();
    const Index n_masters = weights_.size();

    // Read row i / column j of the current matrix into a dense vector. Row
    // access scans all nonzeros (column-major storage has no cheap row view),
    // which is acceptable for the small number of constrained DOFs.
    auto read_row = [&](Index i) {
        Vector<T> row = Vector<T>::Zero(n);
        for (Index k = 0; k < K.outerSize(); ++k)
            for (typename SparseMatrix<T>::InnerIterator it(K, k); it; ++it)
                if (it.row() == i) row(it.col()) = it.value();
        return row;
    };
    auto read_col = [&](Index j) {
        Vector<T> col = Vector<T>::Zero(n);
        for (typename SparseMatrix<T>::InnerIterator it(K, j); it; ++it)
            col(it.row()) = it.value();
        return col;
    };

    // Pull out constant load update using the unmodified columns: F -= K.col(s) * c.
    if (constant_ != T(0)) {
        for (Index k = 0; k < n_slaves; ++k) {
            const Index s = slaves_(k);
            for (typename SparseMatrix<T>::InnerIterator it(K, s); it; ++it)
                F(it.row()) -= it.value() * constant_;
        }
    }

    for (Index k = 0; k < n_slaves; ++k) {
        const Index s = slaves_(k);
        const auto masters = masters_.row(k);

        // Distribute slave load to masters.
        const T F_s = F(s);
        for (Index i = 0; i < n_masters; ++i)
            F(masters(i)) += weights_(i) * F_s;

        // Symmetric condensation. Order matters: col(m) += w·col(s) reads col s
        // after row(m) += w·row(s) has updated K(m,s), which is what produces
        // the w²·K(s,s) cross term of the T^T K T condensation.
        for (Index i = 0; i < n_masters; ++i) {
            const Index m = masters(i);
            const T w = weights_(i);

            const Vector<T> row_s = read_row(s);
            for (Index c = 0; c < n; ++c)
                if (row_s(c) != T(0)) K.coeffRef(m, c) += w * row_s(c);

            const Vector<T> col_s = read_col(s);
            for (Index r = 0; r < n; ++r)
                if (col_s(r) != T(0)) K.coeffRef(r, m) += w * col_s(r);
        }

        // Zero out slave row and column.
        const Vector<T> row_s = read_row(s);
        for (Index c = 0; c < n; ++c)
            if (row_s(c) != T(0)) K.coeffRef(s, c) = T(0);
        const Vector<T> col_s = read_col(s);
        for (Index r = 0; r < n; ++r)
            if (col_s(r) != T(0)) K.coeffRef(r, s) = T(0);

        // Enforce constraint equation: u_s - sum(w_m * u_m) = c.
        for (Index i = 0; i < n_masters; ++i) {
            const Index m = masters(i);
            const T weight = weights_(i);
            K.coeffRef(s, m) = -weight;
            K.coeffRef(m, s) = -weight;
        }
        K.coeffRef(s, s) = T(1);
        F(s) = constant_;
    }

    K.prune(T(0));
}

// === Template Instantiations ========================================================

template class LinearConstraint<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LinearConstraint<float>;
#endif

} // namespace pyck
