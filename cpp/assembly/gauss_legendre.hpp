#ifndef PYCK_GAUSS_LEGENDRE_HPP
#define PYCK_GAUSS_LEGENDRE_HPP

#include <array>
#include <Eigen/Core>

#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Gauss-Legendre quadrature rule (strictly one-dimensional).
/// @tparam T Scalar type
template <std::floating_point T>
class GaussLegendre : public QuadratureRule<T>
{
public:

    /**
     * @brief Default constructor.
     */
    GaussLegendre() = default;

    /**
     * @brief Construct a 1D GaussLegendre rule with the given number of points.
     * 
     * @param num_pts The number of quadrature points.
     */
    explicit GaussLegendre(std::size_t num_pts);

private:

    /**
     * @brief Compute the reference Gauss-Legendre nodes and weights for a given number of points.
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes   Output nodes on [-1, 1].
     * @param weights Output quadrature weights.
     */
    void compute_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const override;

    /**
     * @brief Look up precomputed Gauss-Legendre nodes and weights.
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes   Output nodes on [-1, 1].
     * @param weights Output quadrature weights.
     * @return true if the lookup succeeded, false otherwise.
     */
    bool lookup_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const override;

};

} // namespace pyck

#endif // PYCK_GAUSS_LEGENDRE_HPP
