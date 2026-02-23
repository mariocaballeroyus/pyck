#ifndef PYCK_BSPLINE_HPP
#define PYCK_BSPLINE_HPP

#include <cstddef>
#include <vector>

#include "basis.hpp"

namespace pyck
{

/**
 * @brief B-spline basis functions defined on a one-dimensional parametric space
 */
class BSpline : public Basis
{
public:
    /**
     * @brief Construct a B-spline basis with the given degree and knot vector
     * 
     * @param degree Degree of the B-spline basis functions
     * @param knots Knot vector defining the B-spline basis functions
     */
    BSpline(std::size_t degree, const std::vector<double>& knots)
        : Basis(degree), knots_(knots) {}

    /**
     * @brief Evaluate B-spline basis functions at given parameter values
     * 
     * @param params Parameter values at which to evaluate the basis functions
     * @return A matrix of size (m x n) where m is the number of parameter values and 
     *         n is the number of basis functions
     * 
     *         result = { N_0(u_0) N_1(u_0) ... N_n(u_0)
     *                    N_0(u_1) N_1(u_1) ... N_n(u_1)
     *                    ...      ...      ... ...
     *                    N_0(u_m) N_1(u_m) ... N_n(u_m) }
     */
    Eigen::MatrixXd eval(const Eigen::VectorXd& params) const override;

    /**
     * @brief Evaluate B-spline basis functions and their derivatives at given parameter values
     * 
     * @param params Parameter values at which to evaluate the basis functions
     * @param order Highest order of derivatives to compute
     * @return A vector of matrices, where each matrix corresponds to the basis functions or 
     *         their derivatives
     * 
     *          results[k] = { d^k(N_0)/du^k(u_0) d^k(N_1)/du^k(u_0) ... d^k(N_n)/du^k(u_0)
     *                         d^k(N_0)/du^k(u_1) d^k(N_1)/du^k(u_1) ... d^k(N_n)/du^k(u_1)
     *                         ...                ...                ... ...
     *                         d^k(N_0)/du^k(u_m) d^k(N_1)/du^k(u_m) ... d^k(N_n)/du^k(u_m) }
     * 
     */
    std::vector<Eigen::MatrixXd> eval_derivs(const Eigen::VectorXd& params, 
                                      std::size_t order = 0) const override;

    std::size_t num_basis() const override { return knots_.size() - degree_ - 1; }

    /// @brief Get the knot vector
    const std::vector<double>& knots() const { return knots_; }

private:
    /// @brief Knot vector defining the B-spline basis functions
    std::vector<double> knots_;

    /**
     * @brief Find the knot span index for a given parameter value
     * 
     * @param u Parameter value for which to find the span
     * @return Index of the knot span containing u
     */
    std::size_t find_span(double u) const;
    
    /**
     * @brief Compute the table of basis function values for given parameter values
     * 
     * @param params Parameter values at which to compute the basis functions
     * @return A vector of matrices, where each matrix corresponds to the basis functions of a certain degree
     */
    std::vector<Eigen::MatrixXd> compute_basis_table(const Eigen::VectorXd& params) const;
};

} // namespace pyck

#endif // PYCK_BSPLINE_HPP