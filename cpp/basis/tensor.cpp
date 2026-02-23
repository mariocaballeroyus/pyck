#include "tensor.hpp"

namespace pyck
{

std::size_t TensorProduct::num_basis() const
{
    std::size_t total = 1;
    for (const auto& basis : bases_) {
        // Multiply the number of basis functions in each parametric dimension
        total *= basis->num_basis();
    }
    return total;
}

Eigen::MatrixXd TensorProduct::eval(const Eigen::MatrixXd& params) const
{
    std::size_t num_points = params.rows();
    std::size_t d = dim();

    if (d == 0) return Eigen::MatrixXd(0, 0);

    // Start by evaluating the first parametric dimension
    Eigen::MatrixXd result = bases_[0]->eval(params.col(0));

    // Accumulate the row-wise Kronecker product for all subsequent dimensions
    for (std::size_t i = 1; i < d; ++i) {
        Eigen::MatrixXd mat_i = bases_[i]->eval(params.col(i));
        
        std::size_t cols_res = result.cols();
        std::size_t cols_mat = mat_i.cols();
        
        // The new matrix will have (cols_res * cols_mat) columns
        Eigen::MatrixXd next_result(num_points, cols_res * cols_mat);
        
        for (std::size_t c1 = 0; c1 < cols_res; ++c1) {
            for (std::size_t c2 = 0; c2 < cols_mat; ++c2) {
                // cwiseProduct does an element-by-element multiplication of the two columns
                next_result.col(c1 * cols_mat + c2) = result.col(c1).cwiseProduct(mat_i.col(c2));
            }
        }
        
        // Move the expanded matrix into result for the next iteration
        result = std::move(next_result);
    }

    return result;
}

std::vector<Eigen::MatrixXd> TensorProduct::eval_derivs(const Eigen::MatrixXd& params, 
                                                        const std::vector<std::size_t>& orders) const
{
    std::size_t num_points = params.rows();
    std::size_t d = dim();

    if (d == 0) return {};

    // Get the sequence of derivative matrices for the first dimension
    std::vector<Eigen::MatrixXd> accumulated_results = bases_[0]->eval_derivs(params.col(0), orders[0]);

    // Iteratively compute the tensor product of the derivative combinations
    for (std::size_t i = 1; i < d; ++i) {
        std::vector<Eigen::MatrixXd> mat_i_derivs = bases_[i]->eval_derivs(params.col(i), orders[i]);
        
        std::vector<Eigen::MatrixXd> next_accumulated;
        next_accumulated.reserve(accumulated_results.size() * mat_i_derivs.size());

        // Cross-multiply every derivative order from previous dimensions 
        // with every derivative order of the current dimension
        for (const auto& res_mat : accumulated_results) {
            for (const auto& new_mat : mat_i_derivs) {
                
                std::size_t cols_res = res_mat.cols();
                std::size_t cols_mat = new_mat.cols();
                
                Eigen::MatrixXd combined(num_points, cols_res * cols_mat);
                
                for (std::size_t c1 = 0; c1 < cols_res; ++c1) {
                    for (std::size_t c2 = 0; c2 < cols_mat; ++c2) {
                        combined.col(c1 * cols_mat + c2) = res_mat.col(c1).cwiseProduct(new_mat.col(c2));
                    }
                }
                next_accumulated.push_back(std::move(combined));
            }
        }
        accumulated_results = std::move(next_accumulated);
    }

    return accumulated_results;
}

} // namespace pyck