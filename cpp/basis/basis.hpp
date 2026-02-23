#ifndef PYCK_BASIS_HPP
#define PYCK_BASIS_HPP

#include <cstddef>
#include <vector>

#include<Eigen/Dense>

namespace pyck
{

/**
 * Abstract base class for basis functions defined on a one-dimensional 
 * parametric space
 */
class Basis
{
public:
    /// @brief Virtual destructor
    virtual ~Basis() = default;

    /// @brief Construct a basis with the given degree
    /// @param degree Polynomial degree of the basis functions
    explicit Basis(std::size_t degree) : degree_(degree) {}

    /**
     * @brief Evaluate the basis functions
     * 
     * @param params A vector of parameter values at which to evaluate the basis functions
     * @return A matrix of size (m x n) where m is the number of parameter values and 
     *         n is the number of basis functions.
     * 
     *          result = { N_0(u_0) N_1(u_0) ... N_n(u_0)
     *                     N_0(u_1) N_1(u_1) ... N_n(u_1)
     *                     ...      ...      ... ...
     *                     N_0(u_m) N_1(u_m) ... N_n(u_m) }
     */
    virtual Eigen::MatrixXd eval(const Eigen::VectorXd& params) const = 0;


    /**
     * @brief Evaluate basis functions and their derivatives at given parameter values
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
    virtual std::vector<Eigen::MatrixXd> eval_derivs(const Eigen::VectorXd& params, 
                                                     std::size_t order = 0) const = 0;

    /// @brief Get the degree of the basis functions
    std::size_t degree() const { return degree_; }

    /// @brief Get the number of basis functions
    virtual std::size_t num_basis() const = 0;

protected:
    /// @brief Degree of the basis functions
    std::size_t degree_;
};

} // namespace pyck

#endif // PYCK_BASIS_HPP