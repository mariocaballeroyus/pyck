#include "refinement.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pyck::bspline_kernel
{

// === Knot Insertion =================================================================

template <std::floating_point T>
static Matrix<T> single_boehm_transform(Index p,
                                        const KnotVector<T>& kv,
                                        T u)
{
    const T eps = T(1e-14);
    const Index n_old = kv.size() - p - 1;
    const Index n_new = n_old + 1;
    const Index k = kv.find_span(p, u);

    Matrix<T> M = Matrix<T>::Zero(n_new, n_old);

    for (Index i = 0; i < n_new; ++i)
    {
        if (i + p <= k)
        {
            M(i, i) = T(1);
        }
        else if (i <= k)
        {
            const T denom = kv[i + p] - kv[i];
            const T alpha = (std::abs(denom) > eps)
                          ? (u - kv[i]) / denom : T(0);
            M(i, i - 1) = T(1) - alpha;
            M(i, i)     = alpha;
        }
        else
        {
            M(i, i - 1) = T(1);
        }
    }

    return M;
}

template <std::floating_point T>
std::pair<KnotVector<T>, Matrix<T>>
insert_knot(Index degree, const KnotVector<T>& knots, T u)
{
    Matrix<T> transform = single_boehm_transform<T>(degree, knots, u);
    return {knots.insert_knot(u), std::move(transform)};
}

// === Degree Elevation ===============================================================

template <std::floating_point T>
static Matrix<T> single_knot_removal_transform(Index p,
                                              const KnotVector<T>& kv_before,
                                              T u)
{
    const T eps = T(1e-14);
    const Index n_before = kv_before.size() - p - 1;
    const Index n_after = n_before + 1;
    const Index k = kv_before.find_span(p, u);

    Matrix<T> R = Matrix<T>::Zero(n_before, n_after);

    // Identity prefix: rows 0 .. k-p-1 (the recursion overwrites row k-p).
    for (Index i = 0; i + p < k; ++i) {
        R(i, i) = T(1);
    }

    // Identity suffix: rows k .. n_before-1, R[i, i+1] = 1 (P_i = Q_{i+1}).
    for (Index i = k; i < n_before; ++i) {
        R(i, i + 1) = T(1);
    }

    // Recursive band: writes rows k-1, k-2, ..., k-p using
    // P_{i-1} = (Q_i - alpha_i P_i) / (1 - alpha_i).
    for (Index iter = 0; iter < p; ++iter)
    {
        const Index i = k - iter;
        const T denom = kv_before[i + p] - kv_before[i];
        const T alpha = (std::abs(denom) > eps)
                      ? (u - kv_before[i]) / denom : T(0);
        const T one_minus = T(1) - alpha;
        if (std::abs(one_minus) < eps) continue;

        Eigen::Matrix<T, 1, Eigen::Dynamic> row = -alpha * R.row(i);
        row(i) += T(1);
        R.row(i - 1) = row / one_minus;
    }

    return R;
}

template <std::floating_point T>
static std::pair<KnotVector<T>, Matrix<T>>
single_elevate_step(Index p, const KnotVector<T>& kv)
{
    const Index n_old = kv.size() - p - 1;
    const T u0 = kv.front();
    const T un = kv.back();

    // Enumerate unique interior knots and their multiplicities.
    std::vector<T> u_int;
    std::vector<Index> mult_int;
    {
        Index i = 0;
        while (i < kv.size())
        {
            Index j = i;
            while (j < kv.size() && kv[j] == kv[i]) ++j;
            if (kv[i] > u0 && kv[i] < un)
            {
                u_int.push_back(kv[i]);
                mult_int.push_back(j - i);
            }
            i = j;
        }
    }

    // Step 1: Bezier extraction
    KnotVector<T> kv_bez = kv;
    Matrix<T> E = Matrix<T>::Identity(n_old, n_old);
    for (std::size_t ki = 0; ki < u_int.size(); ++ki)
    {
        const T u = u_int[ki];
        const Index s = mult_int[ki];
        const Index to_insert = (p > s) ? (p - s) : 0;
        for (Index j = 0; j < to_insert; ++j)
        {
            Matrix<T> step = single_boehm_transform<T>(p, kv_bez, u);
            E = step * E;
            kv_bez = kv_bez.insert_knot(u);
        }
    }

    // Step 2: per-segment Bezier elevation
    const Index n_seg = u_int.size() + 1;
    const Index n_bez = kv_bez.size() - p - 1;
    const Index n_bez_elev = n_seg * (p + 1) + 1;

    Matrix<T> B = Matrix<T>::Zero(n_bez_elev, n_bez);
    for (Index seg = 0; seg < n_seg; ++seg)
    {
        const Index off_in  = seg * p;
        const Index off_out = seg * (p + 1);
        B(off_out, off_in) = T(1);
        for (Index i = 1; i <= p; ++i)
        {
            const T frac = static_cast<T>(i) / static_cast<T>(p + 1);
            B(off_out + i, off_in + i - 1) += frac;
            B(off_out + i, off_in + i)     += (T(1) - frac);
        }
        B(off_out + p + 1, off_in + p) = T(1);
    }

    // Elevated-Bezier knot vector (each unique knot's multiplicity +1).
    KnotVector<T> kv_bez_elev = kv_bez.elevate();

    // Step 3: knot removal back to (s_k + 1) at each interior
    KnotVector<T> kv_curr = kv_bez_elev;
    Matrix<T> R = Matrix<T>::Identity(n_bez_elev, n_bez_elev);
    for (std::size_t ki = 0; ki < u_int.size(); ++ki)
    {
        const T u = u_int[ki];
        const Index s = mult_int[ki];
        const Index to_remove = (p > s) ? (p - s) : 0;
        for (Index j = 0; j < to_remove; ++j)
        {
            KnotVector<T> kv_before = kv_curr.drop_knot(u);
            Matrix<T> R_step = single_knot_removal_transform<T>(p + 1, kv_before, u);
            R = R_step * R;
            kv_curr = std::move(kv_before);
        }
    }

    Matrix<T> M = R * B * E;
    return {kv.elevate(), std::move(M)};
}

template <std::floating_point T>
std::pair<KnotVector<T>, Matrix<T>>
elevate_degree(Index degree, const KnotVector<T>& knots)
{
    return single_elevate_step<T>(degree, knots);
}

// === Explicit Instantiations ========================================================

template std::pair<KnotVector<double>, Matrix<double>>
insert_knot<double>(Index, const KnotVector<double>&, double);

template std::pair<KnotVector<double>, Matrix<double>>
elevate_degree<double>(Index, const KnotVector<double>&);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template std::pair<KnotVector<float>, Matrix<float>>
insert_knot<float>(Index, const KnotVector<float>&, float);

template std::pair<KnotVector<float>, Matrix<float>>
elevate_degree<float>(Index, const KnotVector<float>&);
#endif

} // namespace pyck::bspline_kernel
