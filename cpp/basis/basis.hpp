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
     * @brief Evaluate the basis functions and their derivatives up to a given order
     * 
     * @param u A vector of parameter values at which to evaluate the basis functions
     * @param order Highest order of derivatives to compute (0 = values only)
     * @return A vector of matrices, where index k contains the k-th order derivative.
     *         For each matrix, rows correspond to parameter values and columns to
     *         basis functions.
     */
    virtual std::vector<Eigen::MatrixXd> eval(const Eigen::VectorXd& u, 
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