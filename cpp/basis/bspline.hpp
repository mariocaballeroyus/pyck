#ifndef PYCK_BSPLINE_HPP
#define PYCK_BSPLINE_HPP

#include <cstddef>
#include <vector>

#include "basis.hpp"
#include "knot_vector.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief B-spline basis functions defined on a one-dimensional parametric space.ce.
 *
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class BSpline : public Basis<T>
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Construct a B-spline basis with the given degree and knot vector
     *
     * @param degree Degree of the B-spline basis functions
     * @param knots Knot vector defining the B-spline basis functions
     */
    BSpline(Index degree, KnotVector<T> knots)
        : Basis<T>(degree, std::move(knots)) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate non-zero basis functions and mixed partial derivatives for the
     *        tensor product space within a given knot span.
     *
     * @param points      Parameter values on a single knot span.
     * @param span_idx    Knot-span index.
     * @param order       Maximum derivative order.
     * @param uni_results Output buffer, caller-sized to `order + 1`. Inner
     *                    matrices are resized to ((p+1) × m).
     */
    void eval_on_span(const Vector<T>& points, Index span_idx, Index order,
                      std::vector<Matrix<T>>& uni_results) const override;

    // === Refinement =================================================================

    /**
     * @brief Boehm knot insertion (single step).
     *
     * @param u Knot value to insert.
     * @return A pair of (refined basis, transform matrix).
     */
    std::pair<Ptr<Basis<T>>, Matrix<T>> insert_knot(T u) const override;

    /**
     * @brief Degree elevation by one (p-refinement).
     *
     * Implements the textbook Bezier-extraction / Bezier-elevation /
     * knot-removal composition (Piegl-Tiller §5.5). Continuity at each
     * existing internal knot is preserved.
     *
     * @return A pair of (elevated basis, transform matrix).
     */
    std::pair<Ptr<Basis<T>>, Matrix<T>> elevate_degree() const override;

    /// @brief Polymorphic deep copy.
    Ptr<Basis<T>> clone() const override
    { return std::make_shared<BSpline<T>>(this->degree_, this->knots_); }

    // === Factory Methods =============================================================

    /**
     * @brief Create a clamped, uniformly-spaced B-spline basis on [0, 1].
     * @param degree Polynomial degree.
     * @param num_basis Number of basis functions. Must satisfy `num_basis >= degree + 1`.
     * @return A B-spline basis with clamped uniform knots.
     */
    static BSpline<T> clamped_uniform(Index degree, Index num_basis);

};

} // namespace pyck

#endif // PYCK_BSPLINE_HPP
