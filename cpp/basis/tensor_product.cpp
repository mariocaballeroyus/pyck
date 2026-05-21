#include "tensor_product.hpp"

#include "bspline.hpp"
#include "bspline_algorithms.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
TensorProduct<T, d>::TensorProduct(std::array<Ptr<const Basis<T>>, d> bases)
    : bases_(std::move(bases))
{
    for (std::size_t i = 0; i < d; ++i) {
        if (!bases_[i]) {
            throw std::invalid_argument("TensorProduct: "
                                        "Basis pointer is null.");
        }
    }
}

// === Properties =====================================================================

template <std::floating_point T, std::size_t d>
Index TensorProduct<T, d>::num_basis() const
{
    Index total = 1;
    for (std::size_t i = 0; i < d; ++i) {
        total *= bases_[i]->num_basis();
    }
    return total;
}

template <std::floating_point T, std::size_t d>
std::array<Index, d> TensorProduct<T, d>::num_intervals() const
{
    std::array<Index, d> intervals;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = bases_[i]->num_intervals();
    }
    return intervals;
}

template <std::floating_point T, std::size_t d>
Index TensorProduct<T, d>::num_elements() const
{
    Index n = 1;
    for (std::size_t i = 0; i < d; ++i) {
        n *= static_cast<Index>(bases_[i]->knot_vector().non_zero_spans().size());
    }
    return n;
}

template <std::floating_point T, std::size_t d>
std::array<Index, d> TensorProduct<T, d>::decode_element(Index elem_idx) const
{
    std::array<Index, d> spans;
    Index t = elem_idx;
    for (std::size_t i = 0; i < d; ++i) {
        const auto& nz = bases_[i]->knot_vector().non_zero_spans();
        const Index n_i = static_cast<Index>(nz.size());
        spans[i] = static_cast<Index>(nz[t % n_i]);
        t /= n_i;
    }
    return spans;
}

template <std::floating_point T, std::size_t d>
const Basis<T>& TensorProduct<T, d>::basis(Index dir) const
{
    if (dir >= d) {
        throw std::out_of_range("Basis dimension index out of range");
    }
    return *bases_[dir];
}

// === Evaluation =====================================================================

template <std::floating_point T, std::size_t d>
void
TensorProduct<T, d>::eval_on_span(const ColMatrix<T, d>& coords,
                                  const std::array<Index, d>& spans,
                                  Index order,
                                  std::array<std::vector<Matrix<T>>, d>& uni_results,
                                  std::vector<Matrix<T>>& results) const
{
    if (order < 0 || order > MAX_DERIV_ORDER) {
        throw std::invalid_argument("TensorProduct::eval_on_span: "
                                    "order out of supported range.");
    }

    const Index Q = static_cast<Index>(coords.rows());

    std::array<Index, d> n_b{};

    Index K = 1;
    for (std::size_t dim = 0; dim < d; ++dim) {
        uni_results[dim].resize(order + 1);
        bases_[dim]->eval_on_span(coords.col(dim), spans[dim], order, uni_results[dim]);
        n_b[dim] = static_cast<Index>(uni_results[dim][0].rows());
        K *= n_b[dim];
    }

    resize_basis_buffer<T, d>(results, K, Q, order);

    const auto& comps_all = compositions_by_order<d>();

    for (Index k = 0; k <= order; ++k) {
        const auto& compositions = comps_all[k];
        const Index n_slots = static_cast<Index>(compositions.size());

        std::array<Index, d> v{};
        Index r = 0;
        do {
            for (Index m = 0; m < n_slots; ++m) {
                const auto& cc = compositions[m];
                results[k].row(r) = uni_results[0][cc[0]].row(v[0]);
                for (std::size_t i = 1; i < d; ++i)
                    results[k].row(r).array() *= uni_results[i][cc[i]].row(v[i]).array();
                ++r;
            }
        } while (next_lexicographic(v, n_b));
    }
}

template <std::floating_point T, std::size_t d>
std::vector<Matrix<T>>
TensorProduct<T, d>::eval_all(const ColMatrix<T, d>& coords, Index order) const
{
    if (order < 0 || order > MAX_DERIV_ORDER) {
        throw std::invalid_argument("TensorProduct::eval_all: "
                                    "order out of supported range.");
    }

    const Index Q = static_cast<Index>(coords.rows());

    // K = ∏(p_i + 1): same number of active basis functions on every span.
    Index K = 1;
    for (std::size_t dim = 0; dim < d; ++dim) {
        K *= bases_[dim]->degree() + 1;
    }

    std::vector<Matrix<T>> results;
    resize_basis_buffer<T, d>(results, K, Q, order);

    if (Q == 0) return results;

    std::array<std::vector<Matrix<T>>, d> uni_results;
    std::vector<Matrix<T>>                batch_results;
    std::array<Index, d>                  current_spans;

    // Seed current_spans from the first point.
    for (std::size_t dim = 0; dim < d; ++dim) {
        current_spans[dim] = bases_[dim]->find_span(coords(0, dim));
    }

    Index batch_start = 0;

    auto flush_batch = [&](Index end) {
        const Index Qb = end - batch_start;
        ColMatrix<T, d> batch_coords = coords.middleRows(batch_start, Qb);
        eval_on_span(batch_coords, current_spans, order, uni_results, batch_results);
        for (Index k = 0; k <= order; ++k) {
            results[k].middleCols(batch_start, Qb) = batch_results[k];
        }
    };

    for (Index q = 1; q < Q; ++q) {
        std::array<Index, d> new_spans = current_spans;
        bool changed = false;
        for (std::size_t dim = 0; dim < d; ++dim) {
            const T u = coords(q, dim);
            const auto& kv = bases_[dim]->knot_vector();
            // Cache: stay in the current span if u is still within its bounds;
            // only call find_span when crossing a knot boundary.
            if (u < kv[current_spans[dim]] || u >= kv[current_spans[dim] + 1]) {
                new_spans[dim] = bases_[dim]->find_span(u);
                changed = true;
            }
        }
        if (changed) {
            flush_batch(q);
            current_spans = new_spans;
            batch_start   = q;
        }
    }
    flush_batch(Q);

    return results;
}

// === Template Instantiations ========================================================

template class TensorProduct<double, 1>;
template class TensorProduct<double, 2>;
template class TensorProduct<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class TensorProduct<float, 1>;
template class TensorProduct<float, 2>;
template class TensorProduct<float, 3>;
#endif

} // namespace pyck
