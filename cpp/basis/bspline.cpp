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

// === Template Instantiations ========================================================

template class BSpline<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BSpline<float>;
#endif

} // namespace pyck
