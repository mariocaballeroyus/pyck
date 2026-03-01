#ifndef PYCK_GAUSS_LEGENDRE_HPP
#define PYCK_GAUSS_LEGENDRE_HPP

#include <array>
#include <Eigen/Core>

#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Gauss-Legendre quadrature rule.
/// @tparam T Scalar type
/// @tparam d Dimension
template <std::floating_point T, std::size_t d>
class GaussLegendre : public QuadratureRule<T, d>
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Default constructor.
     */
    GaussLegendre() = default;

    /**
     * @brief Construct an isotropic d-dimensional GaussLegendre object.
     * 
     * @param num_pts The number of points for each dimension uniformly.
     */
    explicit GaussLegendre(std::size_t num_pts);

    /**
     * @brief Construct an anisotropic d-dimensional GaussLegendre object.
     * 
     * @param num_pts An array of quadrature dimension points.
     */
    explicit GaussLegendre(const std::array<std::size_t, d>& num_pts);

private:

    // === Utility Methods =============================================================

    /**
     * @brief Compute the reference Gauss-Legendre quadrature rule for a given number of points.
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes The quadrature points.
     * @param weights The quadrature weights.
     */
    void compute_reference(std::size_t num_pts, ColMatrix<T, 1>& nodes, Vector<T>& weights) const override;

    /**
     * @brief Look up the reference Gauss-Legendre quadrature rule for a given number of points.
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes The quadrature points.
     * @param weights The quadrature weights.
     * @return true if the quadrature rule was found, false otherwise.
     */
    bool lookup_reference(std::size_t num_pts, ColMatrix<T, 1>& nodes, Vector<T>& weights) const override;

};

} // namespace pyck

#endif // PYCK_GAUSS_LEGENDRE_HPP
