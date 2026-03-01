#include "bspline.hpp"

namespace pyck
{

template <std::floating_point T>
std::vector<Matrix<T>> BSpline<T>::compute_basis_table(const Vector<T>& params) const
{
    Index num_points = params.size();
    Index num_knots = knots_.size();
    Index num_basis = knots_.num_basis(this->degree_); 

    // The k-th derivative is computed using basis of all lower degrees,
    // (p-1, p-2, ..., p-k), so we need to store them all in a table.
    std::vector<Matrix<T>> table(this->degree_ + 1);
    for (Index d = 0; d <= this->degree_; ++d) {
        // Preallocate with size: num_points x num_basis
        table[d] = Matrix<T>::Zero(num_points, num_basis);
    }

    // Compute basis functions using Cox-de Boor recursion
    for (Index i = 0; i < num_points; ++i) {
        T u = params(i);
        int span = knots_.find_span(this->degree_, u);

        // 0-th degree basis definition
        table[0](i, span) = static_cast<T>(1.0);

        // Recursive computation for degrees 1 .. p
        for (Index d = 1; d <= this->degree_; ++d) {
            for (Index k = 0; k <= d; ++k) {

                // Map the local index k to the global basis function index j
                int j = span - static_cast<int>(d) + static_cast<int>(k); 

                // Skip out-of-range indices
                if (j < 0 || j >= static_cast<int>(num_knots - d - 1)) continue;

                T val = static_cast<T>(0.0);

                // Cox-de Boor: Left term
                T denom_left = knots_[j + d] - knots_[j];
                if (denom_left > static_cast<T>(1e-14)) {
                    val += ((u - knots_[j]) / denom_left) * table[d - 1](i, j);
                }
                
                // Cox-de Boor: Right term
                T denom_right = knots_[j + d + 1] - knots_[j + 1];
                if (denom_right > static_cast<T>(1e-14)) {
                    val += ((knots_[j + d + 1] - u) / denom_right) * table[d - 1](i, j + 1);
                }

                table[d](i, j) = val;
            }
        }
    }
    return table;
}

template <std::floating_point T>
Matrix<T> BSpline<T>::eval(const Vector<T>& params) const
{
    return std::move(compute_basis_table(params)[this->degree_]);
}

template <std::floating_point T>
std::vector<Matrix<T>> BSpline<T>::eval_derivs(const Vector<T>& params, 
                                                     Index order) const
{
    Index num_points = params.size();
    Index num_basis = knots_.num_basis(this->degree_);

    std::vector<Matrix<T>> results(order + 1);
    std::vector<Matrix<T>> table = compute_basis_table(params);
    results[0] = table[this->degree_];

    for (Index k = 1; k <= order; ++k) {
        
        // Derivatives higher than the polynomial degree are strictly zero
        if (k > this->degree_) {
            results[k] = Matrix<T>::Zero(num_points, num_basis);
            continue;
        }

        // Preallocate next table for k-th derivative
        std::vector<Matrix<T>> next_table(this->degree_ + 1);
        
        // Compute derivatives from k .. degree
        for (Index d = k; d <= this->degree_; ++d) {

            // Preallocate table for d-th degree derivatives
            Index current_num_basis = knots_.size() - d - 1;
            next_table[d] = Matrix<T>::Zero(num_points, current_num_basis);
            
            // Apply differenciation rule
            for (Index i = 0; i < current_num_basis; ++i) {

                T left_denom = knots_[i + d] - knots_[i];
                T right_denom = knots_[i + d + 1] - knots_[i + 1];

                // Handle potential division by zero
                if (left_denom > static_cast<T>(1e-14)) {
                    next_table[d].col(i) += (static_cast<T>(d) / left_denom) * table[d - 1].col(i);
                }
                if (right_denom > static_cast<T>(1e-14)) {
                    next_table[d].col(i) -= (static_cast<T>(d) / right_denom) * table[d - 1].col(i + 1);
                }
            }
        }
        
        table = next_table;
        results[k] = table[this->degree_];
    }

    return results;
}

// === Template Instantiations ========================================================

template class BSpline<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BSpline<float>;
#endif

} // namespace pyck
