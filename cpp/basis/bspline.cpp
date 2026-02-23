#include "bspline.hpp"

namespace pyck
{

std::vector<Eigen::MatrixXd> BSpline::compute_basis_table(const Eigen::VectorXd& params) const
{
    std::size_t num_points = params.size();
    std::size_t num_knots = knots_.size();
    std::size_t num_basis = knots_.num_basis(degree_); 

    // The k-th derivative is computed using basis of all lower degrees,
    // (p-1, p-2, ..., p-k), so we need to store them all in a table.
    std::vector<Eigen::MatrixXd> table(degree_ + 1);
    for (std::size_t d = 0; d <= degree_; ++d) {
        // Preallocate with size: num_points x num_basis
        table[d] = Eigen::MatrixXd::Zero(num_points, num_basis);
    }

    // Compute basis functions using Cox-de Boor recursion
    for (std::size_t i = 0; i < num_points; ++i) {
        double u = params(i);
        int span = knots_.find_span(degree_, u);

        // 0-th degree basis definition
        table[0](i, span) = 1.0;

        // Recursive computation for degrees 1 .. p
        for (std::size_t d = 1; d <= degree_; ++d) {
            for (std::size_t k = 0; k <= d; ++k) {

                // Map the local index k to the global basis function index j
                int j = span - d + k; 

                // Skip out-of-range indices
                if (j < 0 || j >= static_cast<int>(num_knots - d - 1)) continue;

                double val = 0.0;

                // Cox-de Boor: Left term
                double denom_left = knots_[j + d] - knots_[j];
                if (denom_left > 1e-14) {
                    val += ((u - knots_[j]) / denom_left) * table[d - 1](i, j);
                }
                
                // Cox-de Boor: Right term
                double denom_right = knots_[j + d + 1] - knots_[j + 1];
                if (denom_right > 1e-14) {
                    val += ((knots_[j + d + 1] - u) / denom_right) * table[d - 1](i, j + 1);
                }

                table[d](i, j) = val;
            }
        }
    }
    return table;
}

Eigen::MatrixXd BSpline::eval(const Eigen::VectorXd& params) const
{
    return std::move(compute_basis_table(params)[degree_]);
}

std::vector<Eigen::MatrixXd> BSpline::eval_derivs(const Eigen::VectorXd& params, 
                                                  std::size_t order) const
{
    std::size_t num_points = params.size();
    std::size_t num_basis = knots_.num_basis(degree_);

    std::vector<Eigen::MatrixXd> results(order + 1);
    std::vector<Eigen::MatrixXd> table = compute_basis_table(params);
    results[0] = table[degree_];

    for (std::size_t k = 1; k <= order; ++k) {
        
        // Derivatives higher than the polynomial degree are strictly zero
        if (k > degree_) {
            results[k] = Eigen::MatrixXd::Zero(num_points, num_basis);
            continue;
        }

        // Preallocate next table for k-th derivative
        std::vector<Eigen::MatrixXd> next_table(degree_ + 1);
        
        // Compute derivatives from k .. degree
        for (std::size_t d = k; d <= degree_; ++d) {

            // Preallocate table for d-th degree derivatives
            std::size_t current_num_basis = knots_.size() - d - 1;
            next_table[d] = Eigen::MatrixXd::Zero(num_points, current_num_basis);
            
            // Apply differenciation rule
            for (std::size_t i = 0; i < current_num_basis; ++i) {

                double left_denom = knots_[i + d] - knots_[i];
                double right_denom = knots_[i + d + 1] - knots_[i + 1];

                // Handle potential division by zero
                if (left_denom > 1e-14) {
                    next_table[d].col(i) += (d / left_denom) * table[d - 1].col(i);
                }
                if (right_denom > 1e-14) {
                    next_table[d].col(i) -= (d / right_denom) * table[d - 1].col(i + 1);
                }
            }
        }
        
        table = next_table;
        results[k] = table[degree_];
    }

    return results;
}

} // namespace pyck