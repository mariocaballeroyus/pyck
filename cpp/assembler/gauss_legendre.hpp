#ifndef PYCK_GAUSS_LEGENDRE_HPP
#define PYCK_GAUSS_LEGENDRE_HPP

#include <Eigen/Core>

#include "quadrature.hpp"

namespace pyck
{


class GaussLegendre : public QuadratureRule
{
public:

    // === Constructors ===============================================================

    GaussLegendre() = default;

    explicit GaussLegendre(std::size_t num_pts);

    // === Properties =================================================================

private:

    // === Utility Methods =============================================================

    static void compute_reference(std::size_t num_pts, Eigen::MatrixXd& nodes, Eigen::VectorXd& weights);

    static bool lookup_reference(std::size_t num_pts, Eigen::MatrixXd& nodes, Eigen::VectorXd& weights);

};

} // namespace pyck

#endif // PYCK_GAUSS_LEGENDRE_HPP