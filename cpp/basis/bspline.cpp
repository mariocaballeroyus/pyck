#include "bspline.hpp"

namespace pyck
{

template <std::floating_point T>
std::vector<Matrix<T>> BSpline<T>::compute_basis_table(const Vector<T>& points,
                                                       Index span) const
{
    Index num_points = points.size();

    // table[d] has (d+1) columns: the non-zero degree-d basis functions
    // at the given span.  Column k corresponds to global index (span - d + k).
    std::vector<Matrix<T>> table(this->degree_ + 1);
    for (Index d = 0; d <= this->degree_; ++d) {
        table[d] = Matrix<T>::Zero(num_points, d + 1);
    }

    // 0-th degree: single non-zero function N_{span,0} = 1
    table[0].col(0).setOnes();

    // Recursive computation for degrees 1 .. p (vectorised over all points)
    for (Index d = 1; d <= this->degree_; ++d) {
        for (Index k = 0; k <= d; ++k) {
            // Global basis function index j = span - d + k
            Index j = span - d + k;

            // Left term:  ((u - xi_j) / (xi_{j+d} - xi_j)) * table[d-1](:, k-1)
            if (k >= 1) 
            {
                T denom = this->knots_[j + d] - this->knots_[j];
                if (denom > static_cast<T>(1e-14)) {
                    table[d].col(k).array() +=
                        ((points.array() - this->knots_[j]) / denom)
                        * table[d - 1].col(k - 1).array();
                }
            }

            // Right term: ((xi_{j+d+1} - u) / (xi_{j+d+1} - xi_{j+1})) * table[d-1](:, k)
            if (k < d) 
            {
                T denom = this->knots_[j + d + 1] - this->knots_[j + 1];
                if (denom > static_cast<T>(1e-14)) {
                    table[d].col(k).array() +=
                        ((this->knots_[j + d + 1] - points.array()) / denom)
                        * table[d - 1].col(k).array();
                }
            }
        }
    }
    return table;
}

template <std::floating_point T>
Matrix<T> BSpline<T>::eval(const Vector<T>& points, Index span) const
{
    return std::move(compute_basis_table(points, span)[this->degree_]);
}

template <std::floating_point T>
std::vector<Matrix<T>> BSpline<T>::eval_derivs(const Vector<T>& points,
                                               Index span,
                                               Index order) const
{
    Index num_points = points.size();
    Index p = this->degree_;

    std::vector<Matrix<T>> results(order + 1);
    std::vector<Matrix<T>> table = compute_basis_table(points, span);
    results[0] = table[p]; // (m × (p+1))

    // Iteratively compute derivatives
    for (Index k = 1; k <= order; ++k)
    {
        // Derivatives higher than the polynomial degree are strictly zero
        if (k > p) {
            results[k] = Matrix<T>::Zero(num_points, p + 1);
            continue;
        }

        std::vector<Matrix<T>> next_table(p + 1);

        // Compute derivatives from degree k .. p
        for (Index d = k; d <= p; ++d)
        {
            // table[d-1] has d columns (local indices 0..d-1).
            // next_table[d] has d+1 columns (local indices 0..d).
            next_table[d] = Matrix<T>::Zero(num_points, d + 1);

            for (Index kl = 0; kl <= d; ++kl)
            {
                // Global basis function index
                Index j = span - d + kl;

                // Left term:  d / denom * table[d-1].col(kl-1)
                if (kl >= 1) 
                {
                    T denom = this->knots_[j + d] - this->knots_[j];
                    if (denom > static_cast<T>(1e-14)) {
                        next_table[d].col(kl) += (static_cast<T>(d) / denom) *
                                                  table[d - 1].col(kl - 1);
                    }
                }

                // Right term: -d / denom * table[d-1].col(kl)
                if (kl < d) 
                {
                    T denom = this->knots_[j + d + 1] - this->knots_[j + 1];
                    if (denom > static_cast<T>(1e-14)) {
                        next_table[d].col(kl) -= (static_cast<T>(d) / denom) *
                                                  table[d - 1].col(kl);
                    }
                }
            }
        }

        table = std::move(next_table);
        results[k] = table[p]; // (m, (p+1))
    }
    return results;
}

template <std::floating_point T>
std::vector<Matrix<T>> BSpline<T>::eval_all(const Vector<T>& points,
                                            Index order) const
{
    Index m = points.size();
    Index n = this->num_basis();
    Index p = this->degree_;

    std::vector<Matrix<T>> results(order + 1);
    for (Index k = 0; k <= order; ++k) {
        results[k] = Matrix<T>::Zero(m, n);
    }

    // Sort points into spans to use the efficient span-local evaluators
    for (Index i = 0; i < m; ++i)
    {
        T u = points[i];
        Index span = this->find_span(u);
        
        Vector<T> pt(1); pt[0] = u;
        auto deriv_vals = this->eval_derivs(pt, span, order);

        for (Index k = 0; k <= order; ++k)
        {
            // The span-local results have p+1 columns. 
            // Local column j corresponds to global index (span - p + j).
            for (Index j = 0; j <= p; ++j)
            {
                results[k](i, span - p + j) = deriv_vals[k](0, j);
            }
        }
    }

    return results;
}

// === Template Instantiations ========================================================

template class BSpline<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BSpline<float>;
#endif

} // namespace pyck
