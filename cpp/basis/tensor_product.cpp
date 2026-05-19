#include "tensor_product.hpp"

#include "bspline.hpp"
#include "evaluation.hpp"

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
TensorProduct<T, d>::eval_on_span(const std::type_identity_t<ColMatrix<T, d>>& coords,
                                  const std::array<Index, d>& spans,
                                  Index order,
                                  Evaluator<T>& ev,
                                  BasisValues<T, d>& out) const
{
    if (order < 0 || order > 3) {
        throw std::invalid_argument("TensorProduct::eval_on_span: "
                                    "order must be in [0, 3].");
    }

    const Index Q = static_cast<Index>(coords.rows());

    // Univariate factors. The Evaluator scratch is shared with the 1D basis
    // evaluations; passing it in lets the caller reuse one allocation across
    // many calls to this function.
    std::array<std::vector<Matrix<T>>, d> uni;
    std::array<Index, d>                  n_b{};

    Index K = 1;
    for (std::size_t dim = 0; dim < d; ++dim) {
        uni[dim].resize(order + 1);
        bases_[dim]->eval_on_span(coords.col(dim), spans[dim], uni[dim], ev);
        n_b[dim] = static_cast<Index>(uni[dim][0].rows());
        K *= n_b[dim];
    }

    // Resize the caller-owned output to (K · n_k) × Q per order. No-op when
    // shape is unchanged — repeated calls in an assembly loop are allocation-free.
    out.reset_for(K, Q, order);
    std::vector<Matrix<T>>& per_order = out.data();

    // Main fill loop
    for (Index q = 0; q < Q; ++q)
    {
        if constexpr (d == 1)
        {
            for (Index k = 0; k <= order; ++k) {
                T*       pk = per_order[k].col(q).data();
                const T* uk = uni[0][k].col(q).data();
                for (Index bu = 0; bu < n_b[0]; ++bu)
                    *pk++ = uk[bu];
            }
        }
        else if constexpr (d == 2)
        {
            T* p0 =                per_order[0].col(q).data();
            T* p1 = (order >= 1) ? per_order[1].col(q).data() : nullptr;
            T* p2 = (order >= 2) ? per_order[2].col(q).data() : nullptr;
            T* p3 = (order >= 3) ? per_order[3].col(q).data() : nullptr;

            const T* u0d =                uni[0][0].col(q).data();
            const T* u1d = (order >= 1) ? uni[0][1].col(q).data() : nullptr;
            const T* u2d = (order >= 2) ? uni[0][2].col(q).data() : nullptr;
            const T* u3d = (order >= 3) ? uni[0][3].col(q).data() : nullptr;
            const T* v0d =                uni[1][0].col(q).data();
            const T* v1d = (order >= 1) ? uni[1][1].col(q).data() : nullptr;
            const T* v2d = (order >= 2) ? uni[1][2].col(q).data() : nullptr;
            const T* v3d = (order >= 3) ? uni[1][3].col(q).data() : nullptr;

            for (Index bu = 0; bu < n_b[0]; ++bu)
            {
                const T u0 =                u0d[bu];
                const T u1 = (order >= 1) ? u1d[bu] : T(0);
                const T u2 = (order >= 2) ? u2d[bu] : T(0);
                const T u3 = (order >= 3) ? u3d[bu] : T(0);

                for (Index bv = 0; bv < n_b[1]; ++bv)
                {
                    const T v0 = v0d[bv];
                    *p0++ = u0 * v0;
                    if (order < 1) continue;

                    const T v1 = v1d[bv];
                    *p1++ = u1 * v0;   // ∂u
                    *p1++ = u0 * v1;   // ∂v
                    if (order < 2) continue;

                    const T v2 = v2d[bv];
                    *p2++ = u2 * v0;   // ∂uu   Voigt slot 0
                    *p2++ = u0 * v2;   // ∂vv   slot 1
                    *p2++ = u1 * v1;   // ∂uv   slot 2
                    if (order < 3) continue;

                    const T v3 = v3d[bv];
                    *p3++ = u3 * v0;   // ∂uuu  sorted-lex slot 0
                    *p3++ = u2 * v1;   // ∂uuv  slot 1
                    *p3++ = u1 * v2;   // ∂uvv  slot 2
                    *p3++ = u0 * v3;   // ∂vvv  slot 3
                }
            }
        }
        else if constexpr (d == 3)
        {
            T* p0 =                per_order[0].col(q).data();
            T* p1 = (order >= 1) ? per_order[1].col(q).data() : nullptr;
            T* p2 = (order >= 2) ? per_order[2].col(q).data() : nullptr;
            T* p3 = (order >= 3) ? per_order[3].col(q).data() : nullptr;

            const T* u0d =                uni[0][0].col(q).data();
            const T* u1d = (order >= 1) ? uni[0][1].col(q).data() : nullptr;
            const T* u2d = (order >= 2) ? uni[0][2].col(q).data() : nullptr;
            const T* u3d = (order >= 3) ? uni[0][3].col(q).data() : nullptr;
            const T* v0d =                uni[1][0].col(q).data();
            const T* v1d = (order >= 1) ? uni[1][1].col(q).data() : nullptr;
            const T* v2d = (order >= 2) ? uni[1][2].col(q).data() : nullptr;
            const T* v3d = (order >= 3) ? uni[1][3].col(q).data() : nullptr;
            const T* w0d =                uni[2][0].col(q).data();
            const T* w1d = (order >= 1) ? uni[2][1].col(q).data() : nullptr;
            const T* w2d = (order >= 2) ? uni[2][2].col(q).data() : nullptr;
            const T* w3d = (order >= 3) ? uni[2][3].col(q).data() : nullptr;

            for (Index bu = 0; bu < n_b[0]; ++bu)
            {
                const T u0 =                u0d[bu];
                const T u1 = (order >= 1) ? u1d[bu] : T(0);
                const T u2 = (order >= 2) ? u2d[bu] : T(0);
                const T u3 = (order >= 3) ? u3d[bu] : T(0);

                for (Index bv = 0; bv < n_b[1]; ++bv)
                {
                    const T v0 =                v0d[bv];
                    const T v1 = (order >= 1) ? v1d[bv] : T(0);
                    const T v2 = (order >= 2) ? v2d[bv] : T(0);
                    const T v3 = (order >= 3) ? v3d[bv] : T(0);

                    for (Index bw = 0; bw < n_b[2]; ++bw)
                    {
                        const T w0 = w0d[bw];
                        *p0++ = u0 * v0 * w0;
                        if (order < 1) continue;

                        const T w1 = w1d[bw];
                        *p1++ = u1 * v0 * w0;   // ∂u
                        *p1++ = u0 * v1 * w0;   // ∂v
                        *p1++ = u0 * v0 * w1;   // ∂w
                        if (order < 2) continue;

                        const T w2 = w2d[bw];
                        *p2++ = u2 * v0 * w0;   // ∂uu   Voigt slot 0
                        *p2++ = u0 * v2 * w0;   // ∂vv   slot 1
                        *p2++ = u0 * v0 * w2;   // ∂ww   slot 2
                        *p2++ = u1 * v1 * w0;   // ∂uv   slot 3
                        *p2++ = u1 * v0 * w1;   // ∂uw   slot 4
                        *p2++ = u0 * v1 * w1;   // ∂vw   slot 5
                        if (order < 3) continue;

                        const T w3 = w3d[bw];
                        *p3++ = u3 * v0 * w0;   // ∂uuu  sorted-lex slot 0
                        *p3++ = u2 * v1 * w0;   // ∂uuv  slot 1
                        *p3++ = u2 * v0 * w1;   // ∂uuw  slot 2
                        *p3++ = u1 * v2 * w0;   // ∂uvv  slot 3
                        *p3++ = u1 * v1 * w1;   // ∂uvw  slot 4
                        *p3++ = u1 * v0 * w2;   // ∂uww  slot 5
                        *p3++ = u0 * v3 * w0;   // ∂vvv  slot 6
                        *p3++ = u0 * v2 * w1;   // ∂vvw  slot 7
                        *p3++ = u0 * v1 * w2;   // ∂vww  slot 8
                        *p3++ = u0 * v0 * w3;   // ∂www  slot 9
                    }
                }
            }
        }
    }
    
}

template <std::floating_point T, std::size_t d>
BasisValues<T, d>
TensorProduct<T, d>::eval(const std::type_identity_t<ColMatrix<T, d>>& coords,
                          Index order) const
{
    if (order < 0 || order > 3) {
        throw std::invalid_argument("TensorProduct::eval: "
                                    "order must be in [0, 3].");
    }

    const Index Q = static_cast<Index>(coords.rows());

    // K = ∏(p_i + 1): same number of active basis functions on every span.
    Index K = 1;
    for (std::size_t dim = 0; dim < d; ++dim) {
        K *= bases_[dim]->degree() + 1;
    }

    BasisValues<T, d> out;
    out.reset_for(K, Q, order);

    // Per-point dispatch: find spans, evaluate one point's worth via the
    // span-local kernel, copy into column q. The scratch Evaluator and the
    // single-point BasisValues buffer are reused across q so allocation is
    // amortized after the first iteration.
    Evaluator<T>            ev;
    BasisValues<T, d>       single;
    ColMatrix<T, d>         single_pt(1, d);
    std::array<Index, d>    spans;

    for (Index q = 0; q < Q; ++q) {
        single_pt.row(0) = coords.row(q);
        for (std::size_t dim = 0; dim < d; ++dim) {
            spans[dim] = bases_[dim]->find_span(coords(q, dim));
        }
        eval_on_span(single_pt, spans, order, ev, single);

        for (Index k = 0; k <= order; ++k) {
            out.data()[k].col(q) = single.data()[k].col(0);
        }
    }

    return out;
}

// === Template Instantiations ========================================================

template class TensorProduct<double, 1>;
template class TensorProduct<double, 2>;
template class TensorProduct<double, 3>;

template class BasisValues<double, 1>;
template class BasisValues<double, 2>;
template class BasisValues<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class TensorProduct<float, 1>;
template class TensorProduct<float, 2>;
template class TensorProduct<float, 3>;

template class BasisValues<float, 1>;
template class BasisValues<float, 2>;
template class BasisValues<float, 3>;
#endif

} // namespace pyck
