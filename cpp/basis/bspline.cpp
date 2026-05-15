#include "bspline.hpp"

#include <algorithm>
#include <cmath>

namespace pyck
{

template <std::floating_point T>
std::vector<Matrix<T>> BSpline<T>::eval_on_span(const Vector<T>& points,
                                                Index span,
                                                Index order) const
{
    const T eps = 1e-14;
    const Index m = points.size();
    const Index p = this->degree_;
    const Index n = order;
    const Index max_k = std::min(n, p);
    const Index stride = p + 1;

    // Preallocate output matrices up to k-th derivatives
    std::vector<Matrix<T>> results(n + 1);
    for (Index k = 0; k <= n; ++k)
        results[k] = Matrix<T>::Zero(m, stride);

    // --- Pass 1: triangle construction for basis values ---
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
        left .col(j).array() = points.array() - this->knots_[span + 1 - j];
        right.col(j).array() = this->knots_[span + j] - points.array();

        // Initialize saved value for this level
        auto saved = fn_col(j, j);
        saved.setZero();

        // Level 0 .. j-1: compute N_{i,j}(u) for i = span-j .. span
        for (Index r = 0; r < j; ++r)
        {
            kd(j, r) = this->knots_[span + r + 1]
                      - this->knots_[span + r + 1 - j];

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

    // --- Pass 2: derivatives ---

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

    return results;
}

template <std::floating_point T>
std::vector<Matrix<T>> BSpline<T>::eval_all(const Vector<T>& points,
                                            Index order) const
{
    const T eps = T(1e-15);
    const Index m = points.size();
    const Index n = this->num_basis();
    const Index p = this->degree_;
    const Index num_knots = this->knots_.size();

    // Preallocate output matrices up to k-th derivatives
    std::vector<Matrix<T>> results(order + 1);
    for (Index k = 0; k <= order; ++k)
        results[k] = Matrix<T>::Zero(m, n);

    for (Index pt = 0; pt < m; ++pt)
    {
        const T u = points[pt];
        const Index n_0 = num_knots - 1;

        // Preallocate table for level 0
        std::vector<ColArray<T, 1>> table(p + 1);
        table[0].resize(n_0);
        table[0].setZero();

        // Initialize table for level 0
        // N_{i,0}(u) = 1 if u_{i} <= u < u_{i+1}
        //              0 otherwise
        for (Index i = 0; i < n_0; ++i)
        {
            if (u >= this->knots_[i] && u < this->knots_[i + 1])
                table[0][i] = T(1);
            // Include right boundary of the last non-zero span
            else if (u == this->knots_[num_knots - 1]
                     && this->knots_[i] < this->knots_[i + 1]
                     && this->knots_[i + 1] == u)
                table[0][i] = T(1);
        }

        // Levels 1 … p: Cox–de Boor recursion
        //
        // N_{i,j}(u) = (u - u_i) / (u_{i+j} - u_i) N_{i,j-1}(u)
        //            + (u_{i+j+1} - u) / (u_{i+j+1} - u_{i+1}) N_{i+1,j-1}(u)
        for (Index j = 1; j <= p; ++j)
        {
            // Preallocate table for level j
            const Index n_j = num_knots - j - 1;
            table[j].resize(n_j);
            table[j].setZero();

            for (Index i = 0; i < n_j; ++i)
            {
                const T d_left  = this->knots_[i + j]     - this->knots_[i];
                const T d_right = this->knots_[i + j + 1] - this->knots_[i + 1];

                if (std::abs(d_left) > eps)
                    table[j][i] += (u - this->knots_[i]) / d_left * table[j - 1][i];

                if (std::abs(d_right) > eps)
                    table[j][i] += (this->knots_[i + j + 1] - u) / d_right * table[j - 1][i + 1];
            }
        }

        // 0th derivative: degree-p basis values
        for (Index i = 0; i < n; ++i)
            results[0](pt, i) = table[p][i];

        const Index max_k = std::min(order, p);

        // Derivatives via α-coefficient recursion
        // N^(k)_{i,p}(u) = p!/(p-k)! · Σ_{j=0}^{k} α_{k,j} · N_{i+j, p-k}(u)
        //
        // α_{0,0} = 1
        // α_{k,0} = α_{k-1,0} / (u_{i+p-k+1} - u_i)
        // α_{k,j} = (α_{k-1,j} - α_{k-1,j-1}) / (u_{i+j+p-k+1} - u_{i+j})
        // α_{k,k} = -α_{k-1,k-1} / (u_{i+p+1} - u_{i+k})
        for (Index i = 0; i < n; ++i)
        {
            ColArray<T, 1> alpha_prev(1);
            alpha_prev(0) = T(1);         // α_{0,·}
            T factor = static_cast<T>(p); // running product p!/(p-k)!

            for (Index k = 1; k <= max_k; ++k)
            {
                ColArray<T, 1> alpha = ColArray<T, 1>::Zero(k + 1);

                // α_{k,0}
                T d = this->knots_[i + p - k + 1] - this->knots_[i];
                alpha[0] = (std::abs(d) > eps)
                         ? alpha_prev[0] / d : T(0);

                // α_{k,j}  for j = 1 … k-1
                for (Index j = 1; j < k; ++j)
                {
                    d = this->knots_[i + j + p - k + 1] - this->knots_[i + j];
                    alpha[j] = (std::abs(d) > eps)
                             ? (alpha_prev[j] - alpha_prev[j - 1]) / d : T(0);
                }

                // α_{k,k}
                d = this->knots_[i + p + 1] - this->knots_[i + k];
                alpha[k] = (std::abs(d) > eps)
                         ? -alpha_prev[k - 1] / d : T(0);

                // Assemble: N^(k)_{i,p}(u) = factor · Σ_j α_{k,j} · table[p-k][i+j]
                T val = T(0);
                for (Index j = 0; j <= k; ++j)
                    val += alpha[j] * table[p - k][i + j];

                results[k](pt, i) = factor * val;

                alpha_prev = std::move(alpha);
                if (k < max_k)
                    factor *= static_cast<T>(p - k);
            }
        }
    }

    return results;
}

template <std::floating_point T>
Vector<T> BSpline<T>::greville_abscissae() const
{
    const Index n = this->num_basis();
    const Index p = this->degree_;
    const auto& knots_data = this->knots_.data();
    
    Vector<T> greville(n);
    for (Index i = 0; i < n; ++i) {
        T sum = 0;
        for (Index j = 1; j <= p; ++j) {
            sum += knots_data[i + j];
        }
        greville[i] = p > 0 ? sum / static_cast<T>(p) : knots_data[i];
    }
    return greville;
}

// === Knot Insertion =================================================================

/**
 * Build the (n+1, n) Boehm transform that maps control points of a degree-p
 * basis with knot vector @p kv to the refined basis obtained by inserting a
 * single copy of @p u.
 */
template <std::floating_point T>
static Matrix<T> single_boehm_transform(Index p,
                                        const KnotVector<T>& kv,
                                        T u)
{
    const T eps = T(1e-14);
    const Index n_old = kv.num_basis(p);
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
KnotInsertion<T> BSpline<T>::insert_knot(T u, Index count) const
{
    const Index p = this->degree_;
    const KnotVector<T>& kv0 = this->knots_;

    if (count == 0)
    {
        return KnotInsertion<T>{
            std::make_shared<BSpline<T>>(p, kv0),
            Matrix<T>::Identity(this->num_basis(), this->num_basis())
        };
    }

    KnotVector<T> kv = kv0;
    Matrix<T> transform = single_boehm_transform<T>(p, kv, u);
    kv = kv.insert(u, 1);

    for (Index step = 1; step < count; ++step)
    {
        Matrix<T> step_transform = single_boehm_transform<T>(p, kv, u);
        transform = step_transform * transform;
        kv = kv.insert(u, 1);
    }

    return KnotInsertion<T>{
        std::make_shared<BSpline<T>>(p, std::move(kv)),
        std::move(transform)
    };
}

// === Degree Elevation ===============================================================

/**
 * Left-inverse of a single Boehm insertion at @p u: maps CPs on
 * @p kv_before.insert(u, 1) back to CPs on @p kv_before. For CPs that
 * actually came from degree-elevation of a multi-segment Bezier, the
 * removal is exact.
 */
template <std::floating_point T>
static Matrix<T> single_knot_removal_transform(Index p,
                                              const KnotVector<T>& kv_before,
                                              T u)
{
    const T eps = T(1e-14);
    const Index n_before = kv_before.num_basis(p);
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

/**
 * Remove one copy of @p u from @p kv (a non-allocating splice).
 */
template <std::floating_point T>
static KnotVector<T> drop_one_knot(const KnotVector<T>& kv, T u)
{
    Vector<T> data(kv.size() - 1);
    bool dropped = false;
    Index out = 0;
    for (Index i = 0; i < kv.size(); ++i)
    {
        if (!dropped && kv[i] == u) { dropped = true; continue; }
        data(out++) = kv[i];
    }
    return KnotVector<T>(std::move(data));
}

/**
 * One degree-elevation step on a degree-p B-spline with knot vector @p kv.
 * Returns the elevated knot vector and the (n_new x n_old) CP transform.
 *
 * Composition: M = R * B * E, where
 *   E -- Bezier extraction (insert each interior to multiplicity p)
 *   B -- per-segment Bezier degree elevation
 *   R -- knot removal back to original multiplicities (each plus one)
 */
template <std::floating_point T>
static std::pair<KnotVector<T>, Matrix<T>>
single_elevate_step(Index p, const KnotVector<T>& kv)
{
    const Index n_old = kv.num_basis(p);
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

    // --- Step 1: Bezier extraction ---
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
            kv_bez = kv_bez.insert(u, 1);
        }
    }

    // --- Step 2: per-segment Bezier elevation ---
    const Index n_seg = u_int.size() + 1;
    const Index n_bez = kv_bez.num_basis(p);          // = n_seg * p + 1
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
    KnotVector<T> kv_bez_elev = kv_bez.elevate(1);

    // --- Step 3: knot removal back to (s_k + 1) at each interior ---
    KnotVector<T> kv_curr = kv_bez_elev;
    Matrix<T> R = Matrix<T>::Identity(n_bez_elev, n_bez_elev);
    for (std::size_t ki = 0; ki < u_int.size(); ++ki)
    {
        const T u = u_int[ki];
        const Index s = mult_int[ki];
        const Index to_remove = (p > s) ? (p - s) : 0;
        for (Index j = 0; j < to_remove; ++j)
        {
            KnotVector<T> kv_before = drop_one_knot<T>(kv_curr, u);
            Matrix<T> R_step = single_knot_removal_transform<T>(p + 1, kv_before, u);
            R = R_step * R;
            kv_curr = std::move(kv_before);
        }
    }

    Matrix<T> M = R * B * E;
    return {kv.elevate(1), std::move(M)};
}

template <std::floating_point T>
DegreeElevation<T> BSpline<T>::elevate_degree(Index count) const
{
    const Index p = this->degree_;
    const Index n = this->num_basis();

    if (count == 0)
    {
        return DegreeElevation<T>{
            std::make_shared<BSpline<T>>(p, this->knots_),
            Matrix<T>::Identity(n, n)
        };
    }

    KnotVector<T> kv = this->knots_;
    Index curr_p = p;
    Matrix<T> transform = Matrix<T>::Identity(n, n);

    for (Index step = 0; step < count; ++step)
    {
        auto [kv_new, step_transform] = single_elevate_step<T>(curr_p, kv);
        transform = step_transform * transform;
        kv = std::move(kv_new);
        ++curr_p;
    }

    return DegreeElevation<T>{
        std::make_shared<BSpline<T>>(curr_p, std::move(kv)),
        std::move(transform)
    };
}

// === Template Instantiations ========================================================

template class BSpline<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BSpline<float>;
#endif

} // namespace pyck
