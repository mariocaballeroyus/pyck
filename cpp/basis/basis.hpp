#ifndef PYCK_BASIS_HPP
#define PYCK_BASIS_HPP

#include <cstddef>
#include <vector>
#include <concepts>

#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Abstract base class for basis functions defined on a one-dimensional 
 *        parametric space
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class Basis
{
public:

    // === Constructors ===============================================================

    /// @brief Virtual destructor
    virtual ~Basis() = default;

    /**
     * @brief Construct a basis with the given degree and knots
     * 
     * @param degree Polynomial degree of the basis functions
     * @param knots Knot vector defining the basis functions
     */
    explicit Basis(Index degree, KnotVector<T> knots) 
    : degree_(degree), knots_(std::move(knots)) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the basis functions
     * 
     * @param points A vector of parameter values at which to evaluate the basis 
     *        functions
     * @return A matrix of size (m, n) where m is the number of parameter values and 
     *         n is the number of basis functions.
     * 
     *          result = { N_0(u_0) N_1(u_0) ... N_n(u_0)
     *                     N_0(u_1) N_1(u_1) ... N_n(u_1)
     *                     ...      ...      ... ...
     *                     N_0(u_m) N_1(u_m) ... N_n(u_m) }
     */
    virtual Matrix<T> eval(const Vector<T>& points) const = 0;

    /**
     * @brief Evaluate basis functions and their derivatives at given parameter values
     * 
     * @param points Parameter values at which to evaluate the basis functions
     * @param order Highest order of derivatives to compute
     * @return A vector of matrices, where each matrix corresponds to the basis 
     *         functions or their derivatives
     * 
     *          results[k] = { d^k(N_0)/du^k(u_0) ... d^k(N_n)/du^k(u_0)
     *                         d^k(N_0)/du^k(u_1) ... d^k(N_n)/du^k(u_1)
     *                         ...                ... ...
     *                         d^k(N_0)/du^k(u_m) ... d^k(N_n)/du^k(u_m) }
     */
    virtual std::vector<Matrix<T>> eval_derivs(const Vector<T>& points, 
                                               Index order = 0) const = 0;

    // === Properties =================================================================

    /// @brief Get the degree of the basis functions
    Index degree() const 
    { return degree_; }

    /// @brief Get the number of basis functions
    virtual Index num_basis() const 
    { return knots_.num_basis(degree_); }

    /// @brief Get the number of parametric intervals (elements) for this basis
    virtual Index num_intervals() const 
    { return knots_.size() - 1; }

    /// @brief Get the knot vector object
    const KnotVector<T>& knot_vector() const 
    { return knots_; }

    /// @brief Get the raw knot values
    const std::vector<T>& knots() const 
    { return knots_.data(); }

protected:

    // === Member Variables ===========================================================

    /// @brief Degree of the basis functions
    Index degree_;

    /// @brief Knot vector defining the basis functions
    KnotVector<T> knots_;

};

} // namespace pyck

#endif // PYCK_BASIS_HPP
