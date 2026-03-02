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
        : BasisType(degree, std::move(knots)) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the (p+1) non-zero B-spline basis functions at given parameter
     *        values within a single knot span.
     * 
     * @param points Parameter values (all assumed to lie in knot span `span`).
     * @param span   Knot-span index (as returned by KnotVector::find_span).
     * @return A matrix of size (m, p+1).  Column k gives N_{span-p+k,p}(u_i).
     */
    Matrix<T> eval(const Vector<T>& points, Index span) const override;

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
    std::vector<Matrix<T>> eval_derivs(const Vector<T>& points,
                                       Index span,
                                       Index order = 0) const override;

private:

    // === Internal Methods =======================================================

    /**
     * @brief Compute the span-local triangular table of basis function values.
     * 
     * @param points Parameter values (all in the given span).
     * @param span   Knot-span index.
     * @return table[d] is an (m × (d+1)) matrix holding the non-zero degree-d
     *         basis values.  Column k corresponds to global index (span − d + k).
     */
    std::vector<Matrix<T>> compute_basis_table(const Vector<T>& points,
                                               Index span) const;

};

} // namespace pyck

#endif // PYCK_BSPLINE_HPP
