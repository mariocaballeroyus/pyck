#include "evaluation.hpp"

#include <algorithm>
#include <cmath>

namespace pyck::bspline_kernel
{

template <std::floating_point T>
void eval_on_span(Index degree,
                  const KnotVector<T>& knots,
                  const Vector<T>& points,
                  Index span_idx,
                  Index order,
                  std::vector<Matrix<T>>& results)
{
    const T eps = 1e-14;
    const Index m = points.size();
    const Index p = degree;
    const Index n = order;
    const Index max_k = std::min(n, p);
    const Index stride = p + 1;

    // Size output to (order+1) zero matrices.
    results.assign(n + 1, Matrix<T>::Zero(m, stride));

    // Pass 1: triangle construction for basis values
    // ndu_fn: Stores the triangular table of basis function values N_{i,p}(u).
    // ndu_kd: Stores knot differences (denominators) for each degree.

    // Preallocate ndu matrices
    Matrix<T> ndu_fn = Matrix<T>::Zero(m, stride * stride);
    ColArray<T, 1> ndu_kd = ColArray<T, 1>::Zero(stride * stride);

    // Helper lambdas for ndu array access
    auto fn_col = [&](Index r, Index j) { return ndu_fn.col(r * stride + j); };
    auto kd     = [&](Index j, Index r) -> T& { return ndu_kd[j * stride + r]; };

    // Initialize Level 0: N_{i,0}(u) = 1 if u_{i} <= u < u_{i+1}
    //                                  0 otherwise
    fn_col(0, 0).setOnes();

    // Precompute left and right arrays for the recursion
    Matrix<T> left (m, stride);
    Matrix<T> right(m, stride);

    // Level 1 … p: Cox–de Boor recursion
    for (Index j = 1; j <= p; ++j)
    {
        // Compute left and right span lengths for this polynomial degree
        left .col(j).array() = points.array() - knots[span_idx + 1 - j];
        right.col(j).array() = knots[span_idx + j] - points.array();

        // Initialize saved value for this level
        auto saved = fn_col(j, j);
        saved.setZero();

        // Level 0 .. j-1: compute N_{i,j}(u) for i = span_idx-j .. span_idx
        for (Index r = 0; r < j; ++r)
        {
            kd(j, r) = knots[span_idx + r + 1]
                      - knots[span_idx + r + 1 - j];

            const T inv = (std::abs(kd(j, r)) > eps)
                        ? T(1) / kd(j, r)
                        : T(0);

            auto prev = fn_col(r, j - 1);
            auto dest = fn_col(r, j);

            dest.noalias()  = saved + right.col(r + 1).cwiseProduct(prev) * inv;
            saved.noalias() = left.col(j - r).cwiseProduct(prev) * inv;
        }
    }

    // Load degree-p basis values
    for (Index j = 0; j <= p; ++j)
        results[0].col(j) = fn_col(j, p);

    // If no derivatives requested, we're done
    if (n == 0)
        return;

    // Pass 2: derivatives

    // Level 0 .. p: compute derivatives of N_{i,j}(u)
    for (Index r = 0; r <= p; ++r)
    {
        RowArray<T, 2> a = RowArray<T, 2>::Zero(2, stride);
        Index s1 = 0, s2 = 1;
        a(0, 0) = T(1);

        // Level 1 .. max_k: compute derivatives of N_{i,j}^{(k)}(u)
        for (Index k = 1; k <= max_k; ++k)
        {
            int rk = static_cast<int>(r) - static_cast<int>(k);
            int pk = static_cast<int>(p) - static_cast<int>(k);

            auto d = results[k].col(r);
            d.setZero();

            if (r >= k)
            {
                a(s2, 0) = a(s1, 0) / kd(pk + 1, rk);
                d.array() += a(s2, 0) * fn_col(rk, pk).array();
            }

            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (static_cast<int>(r) - 1 <= pk)
                   ? static_cast<int>(k) - 1
                   : static_cast<int>(p) - static_cast<int>(r);

            for (int j = j1; j <= j2; ++j)
            {
                a(s2, j) = (a(s1, j) - a(s1, j - 1)) / kd(pk + 1, rk + j);
                d.array() += a(s2, j) * fn_col(rk + j, pk).array();
            }

            if (static_cast<int>(r) <= pk)
            {
                a(s2, k) = -a(s1, k - 1) / kd(pk + 1, r);
                d.array() += a(s2, k) * fn_col(r, pk).array();
            }

            std::swap(s1, s2);
        }
    }

    // Multiply through by the correct factors p!/(p-k)!
    T r_factor = static_cast<T>(p);
    for (Index k = 1; k <= max_k; ++k)
    {
        results[k] *= r_factor;
        r_factor *= static_cast<T>(p - k);
    }
}

template <std::floating_point T>
Vector<T> greville_abscissae(Index degree, const KnotVector<T>& knots)
{
    const Index p = degree;
    const Index n = knots.size() - p - 1;
    const auto& knots_data = knots.data();

    // Preallocate Greville abscissae
    Vector<T> greville(n);

    // Loop over basis functions
    for (Index i = 0; i < n; ++i)
    {
        // Compute the sum of the interior knots
        T sum = 0;
        for (Index j = 1; j <= p; ++j)
        {
            sum += knots_data[i + j];
        }

        // Compute the Greville abscissa for basis function N_{i,p}
        greville[i] = p > 0 ? sum / static_cast<T>(p) : knots_data[i];
    }
    return greville;
}

// === Explicit Instantiations ========================================================

template void eval_on_span<double>(Index, const KnotVector<double>&,
                                   const Vector<double>&, Index, Index,
                                   std::vector<Matrix<double>>&);
template Vector<double> greville_abscissae<double>(Index, const KnotVector<double>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template void eval_on_span<float>(Index, const KnotVector<float>&,
                                  const Vector<float>&, Index, Index,
                                  std::vector<Matrix<float>>&);
template Vector<float> greville_abscissae<float>(Index, const KnotVector<float>&);
#endif

} // namespace pyck::bspline_kernel
