#ifndef PYCK_BSPLINE_HPP
#define PYCK_BSPLINE_HPP

#include <cstddef>
#include <vector>

#include "basis.hpp"
#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief B-spline basis functions defined on a one-dimensional parametric space
/// @tparam T Scalar type
template <std::floating_point T = double>
class BSpline : public Basis<T>
{

public:

    // === Constructors ===============================================================

    /// @brief Default constructor
    BSpline() = default;

    /**
     * @brief Construct a B-spline basis with the given degree and knot vector
     * 
     * @param degree Degree of the B-spline basis functions
     * @param knots Knot vector defining the B-spline basis functions
     */
    BSpline(Index degree, KnotVector<T> knots)
        : Basis<T>(degree), knots_(std::move(knots)) {}

    // === Evaluation =================================================================

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
    Matrix<T> eval(const Vector<T>& params) const override;

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
     */
    std::vector<Matrix<T>> eval_derivs(const Vector<T>& params, 
                                        Index order = 0) const override;

    // === Properties =================================================================

    Index num_basis() const override { return knots_.num_basis(this->degree_); }

    /// @brief Get the knot vector object
    const KnotVector<T>& knot_vector() const { return knots_; }

    /// @brief Get the raw knot values (backward-compatible)
    const std::vector<T>& knots() const { return knots_.data(); }

private:

    // === Internal Methods =======================================================

    /**
     * @brief Compute the table of basis function values for given parameter values
     * 
     * @param params Parameter values at which to compute the basis functions
     * @return A vector of matrices, where each matrix corresponds to the basis functions of a certain degree
     */
    std::vector<Matrix<T>> compute_basis_table(const Vector<T>& params) const;

    // === Member Variables =======================================================

    /// @brief Knot vector defining the B-spline basis functions
    KnotVector<T> knots_;

};

} // namespace pyck

#endif // PYCK_BSPLINE_HPP
