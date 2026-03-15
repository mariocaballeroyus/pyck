#ifndef PYCK_GAUSS_LEGENDRE_HPP
#define PYCK_GAUSS_LEGENDRE_HPP

#include <array>
#include <Eigen/Core>

#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief d-dimensional Gauss-Legendre quadrature rule.
 * 
 * For d > 1, this class automatically constructs an isotropic tensor product 
 * of a 1D Gauss-Legendre rule with the specified number of points in each dimension.
 * 
 * @tparam T Scalar floating-point type
 * @tparam d Parametric dimension (default = 1)
 */
template <std::floating_point T, std::size_t d>
class GaussLegendre : public QuadratureRule<T, d>
{
public:

    /**
     * @brief Default constructor.
     */
    GaussLegendre() = default;

    /**
     * @brief Construct a d-dimensional Gauss-Legendre rule with the given number 
     *        of points per dimension.
     * 
     * Internally forms an isotropic tensor product of 1D Gauss-Legendre rules.
     * 
     * @param num_pts The number of quadrature points to use in each dimension.
     */
    explicit GaussLegendre(std::size_t num_pts);

};

/**
 * @brief Partial specialization for one-dimensional Gauss-Legendre quadrature.
 * 
 * Provides the core logic for computing or looking up Gauss nodes and weights 
 * on the reference interval [-1, 1].
 * 
 * @tparam T Scalar floating-point type
 */
template <std::floating_point T>
class GaussLegendre<T, 1> : public QuadratureRule<T, 1>
{
public:
    /// @brief Default constructor.
    GaussLegendre() = default;

    /**
     * @brief Construct a 1D Gauss-Legendre rule with the given number of points.
     * 
     * @param num_pts The number of quadrature points to use.
     */
    explicit GaussLegendre(std::size_t num_pts);

protected:
    /**
     * @brief Compute the reference Gauss-Legendre nodes and weights.
     * 
     * Uses the Golub-Welsch algorithm (eigenvalues of the Jacobi matrix).
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes   Output nodes on [-1, 1] (matrix rows=Q, cols=1).
     * @param weights Output quadrature weights as a vector (size=Q).
     */
    void compute_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const override;

    /**
     * @brief Look up precomputed Gauss-Legendre nodes and weights.
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes   Output nodes on [-1, 1] (matrix rows=Q, cols=1).
     * @param weights Output quadrature weights as a vector (size=Q).
     * @return True if the lookup succeeded, false otherwise.
     */
    bool lookup_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const override;

};

} // namespace pyck

#endif // PYCK_GAUSS_LEGENDRE_HPP
