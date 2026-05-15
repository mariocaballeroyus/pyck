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
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class NURBS : public Basis<T>
{
public:

    using BasisType = Basis<T>;

    // === Constructors ===============================================================

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
    NURBS(Index degree, KnotVector<T> knots, Vector<T> weights);

    // === Evaluation =================================================================

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

    // === Refinement =================================================================

    /**
     * @brief Knot insertion (Piegl-Tiller §5.3 rational variant).
     *
     * The transform applies directly to unweighted 3D control points; the
     * refined weights are baked into the returned NURBS basis so the patch
     * invariant (unweighted control points) is preserved.
     */
    KnotInsertion<T> insert_knot(T u, Index count = 1) const override;

    /**
     * @brief Degree elevation (rational variant).
     *
     * Runs the underlying B-spline elevation on the homogeneous CPs; new
     * weights are baked into the returned NURBS basis so the transform
     * applies directly to unweighted control points.
     */
    DegreeElevation<T> elevate_degree(Index count = 1) const override;

    // === Properties =================================================================

    /// @brief Per-basis-function weights.
    const Vector<T>& weights() const { return weights_; }

    /// @brief Underlying B-spline basis (without rational weighting).
    const BSpline<T>& bspline() const { return bspline_; }

private:

    /// @brief Underlying polynomial B-spline basis (same degree and knots).
    BSpline<T> bspline_;

    /// @brief Per-basis-function weights, indexed 0 .. num_basis - 1.
    Vector<T> weights_;
};

} // namespace pyck

#endif // PYCK_NURBS_HPP
