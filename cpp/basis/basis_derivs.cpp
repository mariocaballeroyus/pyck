#include "basis_derivs.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
BasisDerivs<T, d>
eval_basis(const TensorProduct<T, d>& tp,
           const std::type_identity_t<ColMatrix<T, d>>& coords,
           const std::array<Index, d>& spans,
           Index order)
{
    if (order > 3) {
        throw std::invalid_argument("eval_basis: order > 3 not supported.");
    }

    const Index Q = static_cast<Index>(coords.rows());

    // Per-direction univariate derivatives 0..order.
    // uni[i][k] is (Q × (p_i + 1)): k-th deriv of basis i at coords.col(i).
    std::array<std::vector<Matrix<T>>, d> uni;
    std::array<Index, d>                  n_b{};
    Index K = 1;
    for (std::size_t i = 0; i < d; ++i) {
        tp.basis(i).eval_on_span(coords.col(i), spans[i], order, uni[i]);
        n_b[i] = static_cast<Index>(uni[i][0].cols());
        K *= n_b[i];
    }

    std::vector<Matrix<T>> per_order(order + 1);

    for (Index k = 0; k <= order; ++k)
    {
        const auto dirs_list = enumerate_direction_indices<d>(k);
        const Index n_k = static_cast<Index>(dirs_list.size());
        per_order[k].resize(K * n_k, Q);

        for (Index packed = 0; packed < n_k; ++packed)
        {
            // Convert direction-index tuple → per-direction order tuple a.
            std::array<Index, d> a{};
            for (Index dir : dirs_list[packed]) ++a[dir];

            for (Index basis_idx = 0; basis_idx < K; ++basis_idx)
            {
                // Decompose basis_idx assuming dim-0-OUTER, dim-(d-1)-INNER
                // (matches existing TensorProduct::eval_on_span column layout
                // and DofMapper::get_element_dofs iteration).
                std::array<Index, d> bi{};
                Index rem = basis_idx;
                for (std::size_t dim_rev = 0; dim_rev < d; ++dim_rev) {
                    const std::size_t dim = d - 1 - dim_rev;
                    bi[dim] = rem % n_b[dim];
                    rem /= n_b[dim];
                }

                const Index row = basis_idx * n_k + packed;

                // Vectorised over q: row_vec(q) = prod_dim uni[dim][a[dim]](q, bi[dim])
                Vector<T> row_vec = uni[0][a[0]].col(bi[0]);
                for (std::size_t dim = 1; dim < d; ++dim) {
                    row_vec.array() *= uni[dim][a[dim]].col(bi[dim]).array();
                }
                per_order[k].row(row) = row_vec.transpose();
            }
        }
    }

    return BasisDerivs<T, d>(std::move(per_order));
}

// === Template Instantiations ========================================================

template BasisDerivs<double, 1> eval_basis<double, 1>(
    const TensorProduct<double, 1>&, const ColMatrix<double, 1>&,
    const std::array<Index, 1>&, Index);
template BasisDerivs<double, 2> eval_basis<double, 2>(
    const TensorProduct<double, 2>&, const ColMatrix<double, 2>&,
    const std::array<Index, 2>&, Index);
template BasisDerivs<double, 3> eval_basis<double, 3>(
    const TensorProduct<double, 3>&, const ColMatrix<double, 3>&,
    const std::array<Index, 3>&, Index);

template class BasisDerivs<double, 1>;
template class BasisDerivs<double, 2>;
template class BasisDerivs<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template BasisDerivs<float, 1> eval_basis<float, 1>(
    const TensorProduct<float, 1>&, const ColMatrix<float, 1>&,
    const std::array<Index, 1>&, Index);
template BasisDerivs<float, 2> eval_basis<float, 2>(
    const TensorProduct<float, 2>&, const ColMatrix<float, 2>&,
    const std::array<Index, 2>&, Index);
template BasisDerivs<float, 3> eval_basis<float, 3>(
    const TensorProduct<float, 3>&, const ColMatrix<float, 3>&,
    const std::array<Index, 3>&, Index);

template class BasisDerivs<float, 1>;
template class BasisDerivs<float, 2>;
template class BasisDerivs<float, 3>;
#endif

} // namespace pyck
