#ifndef PYCK_NURBS_HPP
#define PYCK_NURBS_HPP

#include <vector>

#include "basis.hpp"
#include "bspline.hpp"
#include "knots.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief NURBS (Non-Uniform Rational B-Spline) basis on a 1D parametric space.
 *
 * The rational shape functions are
 *
 *      R_{i,p}(u) = w_i N_{i,p}(u) / W(u),     W(u) = Σ_j w_j N_{j,p}(u),
 *
 * where N_{i,p} are the underlying B-spline functions of degree p on the
 * given knot vector and {w_i} are user-supplied positive weights (one per
 * basis function).
 *
 * Evaluation returns the rational functions R_{i,p} and their parametric
 * derivatives, computed from the B-spline derivatives via the standard
 * quotient-rule recurrence (see Piegl & Tiller, "The NURBS Book", §4.2).
 * This means a Patch using a NURBS basis can compute geometry as
 * x(u) = R(u) · P with ordinary (un-weighted) 3D control points.
 *
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class NURBS : public Basis<T>
{
public:

    using BasisType = Basis<T>;

    /// @brief Default constructor
    NURBS() = default;

    /**
     * @brief Construct a NURBS basis from a degree, knot vector and weights.
     *
     * @param degree   Polynomial degree of the underlying B-spline.
     * @param knots    Knot vector (must support num_basis(degree) basis functions).
     * @param weights  Per-basis-function weights (size = num_basis(degree)),
     *                 each strictly positive.
     *
     * @throws std::invalid_argument if the weight count does not match the
     *         number of basis functions, or if any weight is non-positive.
     */
    NURBS(Index degree, KnotVector<T> knots, std::vector<T> weights);

    /**
     * @brief Evaluate the (p+1) non-zero rational basis functions and their
     *        derivatives at the given parameter values within a single span.
     */
    std::vector<Matrix<T>> eval_on_span(const Vector<T>& points,
                                       Index span,
                                       Index order = 0) const override;

    /**
     * @brief Evaluate all rational basis functions and their derivatives at
     *        the given parameter values.
     */
    std::vector<Matrix<T>> eval_all(const Vector<T>& points,
                                    Index order = 0) const override;

    /**
     * @brief Greville abscissae of the NURBS basis (knot averages — identical
     *        to the underlying B-spline, independent of the weights).
     */
    Vector<T> greville_abscissae() const override;

    /// @brief Per-basis-function weights.
    const std::vector<T>& weights() const { return weights_; }

    /// @brief Underlying B-spline basis (without rational weighting).
    const BSpline<T>& bspline() const { return bspline_; }

private:

    /// @brief Underlying polynomial B-spline basis (same degree and knots).
    BSpline<T> bspline_;

    /// @brief Per-basis-function weights, indexed 0 .. num_basis - 1.
    std::vector<T> weights_;
};

} // namespace pyck

#endif // PYCK_NURBS_HPP
