#ifndef PYCK_NURBS_HPP
#define PYCK_NURBS_HPP

#include <vector>

#include "basis.hpp"
#include "knot_vector.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief NURBS (Non-Uniform Rational B-Spline) basis on a 1D parametric space.
 *
 * @details The basis holds no mutable state; concurrent evaluation from
 *          multiple threads is safe provided each thread owns its own
 *          Cox-de-Boor scratch (allocated on the stack inside eval_on_span).
 *
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class NURBS : public Basis<T>
{
public:

    // === Constructors ===============================================================

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
     * @brief Evaluate the non-zero rational basis functions and their
     *        derivatives at given parameter values within a single knot span.
     *
     * @details Evaluates the underlying B-spline derivatives via Cox-de-Boor
     *          plus the derivative recurrence, then applies the rational
     *          quotient rule to obtain R_i^{(k)}.
     *
     * @param points      Parameter values (all assumed to lie in the same knot span).
     * @param span_idx    Knot-span index (as returned by KnotVector::find_span).
     * @param uni_results Output buffer; the caller must size it to `order + 1`,
     *                    where `order` is the maximum derivative order to compute.
     *                    Inner matrices are sized to (m × (p+1)) on return.
     */
    void eval_on_span(const Vector<T>& points,
                      Index span_idx, Index order,
                      std::vector<Matrix<T>>& uni_results) const override;

    // === Refinement =================================================================

    /**
     * @brief Knot insertion. The refined weights are baked into the returned NURBS basis.
     * 
     *
     * @param u Knot value to insert.
     * @return A pair of (refined basis, transform matrix).
     */
    std::pair<Ptr<Basis<T>>, Matrix<T>> insert_knot(T u) const override;

    /**
     * @brief Degree elevation by one (rational variant).
     *
     * New weights are baked into the returned NURBS basis.
     *
     * @return A pair of (elevated basis, transform matrix).
     */
    std::pair<Ptr<Basis<T>>, Matrix<T>> elevate_degree() const override;

    // === Properties =================================================================

    /// @brief Per-basis-function weights.
    const Vector<T>& weights() const { return weights_; }

    // === Factory Methods =============================================================

    /**
     * @brief Create a clamped, uniformly-spaced NURBS basis on [0, 1].
     * @param degree Polynomial degree.
     * @param num_basis Number of basis functions. Must satisfy `num_basis >= degree + 1`.
     * @param weights Per-basis-function weights (length `num_basis`), each strictly positive.
     * @return A NURBS basis with clamped uniform knots.
     */
    static NURBS<T> clamped_uniform(Index degree, Index num_basis, const Vector<T>& weights);

private:

    /// @brief Per-basis-function weights.
    Vector<T> weights_;

};

} // namespace pyck

#endif // PYCK_NURBS_HPP
