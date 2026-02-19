#include "bspline.hpp"

namespace pyck
{

std::vector<Eigen::MatrixXd> evaluate_bspline(std::size_t degree,
                                              const std::vector<double>& knots,
                                              const Eigen::VectorXd& u,
                                              std::size_t order)
{
    std::size_t num_points = u.size();
    std::size_t num_knots = knots.size();
    std::size_t num_basis = num_knots - degree - 1; 

    // --- Compute basis functions table --------------------------------------------

    std::vector<Eigen::MatrixXd> table(degree + 1);
    for (std::size_t d = 0; d <= degree; ++d) {
        table[d] = Eigen::MatrixXd::Zero(num_points, num_knots - d - 1);
    }

    auto find_span = [&](double u_val) -> int {
        int n = num_knots - degree - 2;
        if (u_val >= knots[n + 1]) return n;
        if (u_val <= knots[degree]) return degree;
        auto it = std::upper_bound(knots.begin() + degree, knots.begin() + n + 1, u_val);
        return std::distance(knots.begin(), it) - 1;
    };

    for (std::size_t i = 0; i < num_points; ++i) {
        double current_u = u(i);
        int span = find_span(current_u);
        table[0](i, span) = 1.0;

        for (std::size_t d = 1; d <= degree; ++d) {
            for (std::size_t k = 0; k <= d; ++k) {
                int j = span - d + k; 
                if (j < 0 || j >= static_cast<int>(num_knots - d - 1)) continue;

                double val = 0.0;
                double denom_left = knots[j + d] - knots[j];
                if (denom_left > 1e-14) {
                    val += ((current_u - knots[j]) / denom_left) * table[d - 1](i, j);
                }
                
                double denom_right = knots[j + d + 1] - knots[j + 1];
                if (denom_right > 1e-14) {
                    val += ((knots[j + d + 1] - current_u) / denom_right) * table[d - 1](i, j + 1);
                }
                table[d](i, j) = val;
            }
        }
    }

    // Allocate results vector: indices 0 through order
    std::vector<Eigen::MatrixXd> results(order + 1);
    results[0] = table[degree]; // The standard basis evaluation
    
    // If no derivatives were requested, we are done!
    if (order == 0) return results;

    // --- Compute 1 .. order derivatives -------------------------------------------

    std::vector<Eigen::MatrixXd> D = table;

    for (std::size_t k = 1; k <= order; ++k) {
        
        // Derivatives higher than the polynomial degree are strictly zero
        if (k > degree) {
            results[k] = Eigen::MatrixXd::Zero(num_points, num_basis);
            continue;
        }

        std::vector<Eigen::MatrixXd> next_D(degree + 1);
        
        for (std::size_t d = k; d <= degree; ++d) {
            std::size_t current_num_basis = knots.size() - d - 1;
            next_D[d] = Eigen::MatrixXd::Zero(num_points, current_num_basis);
            
            for (std::size_t i = 0; i < current_num_basis; ++i) {
                double left_denom = knots[i + d] - knots[i];
                double right_denom = knots[i + d + 1] - knots[i + 1];

                if (left_denom > 1e-14) {
                    next_D[d].col(i) += (d / left_denom) * D[d - 1].col(i);
                }
                if (right_denom > 1e-14) {
                    next_D[d].col(i) -= (d / right_denom) * D[d - 1].col(i + 1);
                }
            }
        }
        
        D = next_D;
        results[k] = D[degree];
    }

    return results;
}

std::vector<Eigen::MatrixXd> BSpline::eval(const Eigen::VectorXd& u, 
                                               std::size_t order) const
{
    return evaluate_bspline(degree_, knots_, u, order);
}

} // namespace pyck