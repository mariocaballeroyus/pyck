#ifndef PYCK_BSPLINE_HPP
#define PYCK_BSPLINE_HPP

#include <cstddef>
#include <vector>

#include "basis.hpp"

namespace pyck
{

/**
 * @brief Evaluate B-spline basis functions and their derivatives at given parameter values
 * 
 * @param degree Polynomial degree of the B-Spline basis functions
 * @param knots Knot vector defining the B-spline basis functions
 * @param u Parameter values at which to evaluate the basis functions
 * @param order Highest order of derivatives to compute
 * @return A vector of matrices, where each matrix corresponds to the basis functions or 
 *         their derivatives
 */
std::vector<Eigen::MatrixXd> evaluate_bspline(std::size_t degree,
                                              const std::vector<double>& knots,
                                              const Eigen::VectorXd& u,
                                              std::size_t order);

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
     * @brief Evaluate the B-spline basis functions and their derivatives up to a given order.
     * 
     * @param u Parameter values at which to evaluate the basis functions.
     * @param order Highest order of derivatives to compute. 
     *              If 0, only the basis functions themselves are evaluated.
     * @return A vector of matrices, where index 'k' contains the k-th order derivative.
     *         For each matrix, rows correspond to parameter values and columns to basis functions.
     */
    std::vector<Eigen::MatrixXd> eval(const Eigen::VectorXd& u, 
                                      std::size_t order = 0) const override;

    std::size_t num_basis() const override { return knots_.size() - degree_ - 1; }

    /// @brief Get the knot vector
    const std::vector<double>& knots() const { return knots_; }

private:
    /// @brief Knot vector defining the B-spline basis functions
    std::vector<double> knots_;
};

} // namespace pyck

#endif // PYCK_BSPLINE_HPP