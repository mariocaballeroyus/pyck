#ifndef PYCK_BSPLINE_HPP
#define PYCK_BSPLINE_HPP

#include <cstddef>
#include <vector>

#include "basis.hpp"
#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief B-spline basis functions defined on a one-dimensional parametric space
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class BSpline : public Basis<T>
{
public:

    using BasisType = Basis<T>;

    /// @brief Default constructor
    BSpline() = default;

    /**
     * @brief Construct a B-spline basis with the given degree and knot vector
     * 
     * @param degree Degree of the B-spline basis functions
     * @param knots Knot vector defining the B-spline basis functions
     */
    BSpline(Index degree, KnotVector<T> knots)
        : BasisType(degree, std::move(knots)) {}

    /**
     * @brief Evaluate the (p+1) non-zero B-spline basis functions and their
     *        derivatives at given parameter values within a single knot span.
     * 
     * @param points Parameter values (all assumed to lie in knot span `span`).
     * @param span   Knot-span index (as returned by KnotVector::find_span).
     * @param order  Highest order of derivatives to compute.
     * @return A vector of (order+1) matrices, each of size (m, p+1).
     *         results[k](i, j) = d^k N_{span-p+j,p} / du^k (points[i])
     */
    std::vector<Matrix<T>> eval_on_span(const Vector<T>& points,
                                       Index span,
                                       Index order = 0) const override;

    /**
     * @brief Evaluate all basis functions and their derivatives at given parameter values.
     * 
     * @param points Parameter values.
     * @param order  Highest order of derivatives to compute (default 0).
     * @return A vector of (order+1) matrices, each of size (m, n).
     */
    std::vector<Matrix<T>> eval_all(const Vector<T>& points,
                                    Index order = 0) const override;
};

} // namespace pyck

#endif // PYCK_BSPLINE_HPP
