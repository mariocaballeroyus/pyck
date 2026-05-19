#ifndef PYCK_BASIS_HPP
#define PYCK_BASIS_HPP

#include <cstddef>
#include <vector>
#include <concepts>

#include "evaluation.hpp"
#include "knot_vector.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Abstract base class for basis functions defined on a one-dimensional
 *        parametric space (internal use—derive BSpline or NURBS instead).
 *
 * @tparam T Scalar type
 */
template <std::floating_point T = double>
class Basis
{
public:

    // === Constructors ===============================================================

    /// @brief Virtual destructor
    virtual ~Basis() = default;

    /**
     * @brief Construct a basis with the given degree and knots
     *
     * @param degree Polynomial degree of the basis functions
     * @param knots Knot vector defining the basis functions
     */
    explicit Basis(Index degree, KnotVector<T> knots)
        : degree_(degree),
          knots_(std::move(knots)) {}

    // === Evaluation =================================================================

    /**
     * @brief Evaluate the non-zero (p+1) basis functions and their derivatives
     *        at given parameter values within a single knot span.
     *
     * @details The caller supplies an `Evaluator` that owns the scratch buffers;
     *          this method resizes them as needed and writes into `results`.
     *          The maximum derivative order is taken to be `results.size() - 1`,
     *          so the caller must size `results` to `order + 1` before calling.
     *          Each `results[k]` is resized to **(degree+1) × points.size()**
     *          column-major: column q holds the N = p+1 active basis values
     *          (k-th derivative) at parameter `points[q]`. This q-major layout
     *          matches the per-order packed buffer produced by
     *          `TensorProduct::eval_on_span` so downstream code reads
     *          contiguous N-vectors per quadrature point.
     *
     * @param points    Parameter values (all assumed to lie in the same knot span).
     * @param span_idx  Knot-span index (as returned by KnotVector::find_span).
     * @param results   Output buffer, pre-sized to `order + 1`. Each entry is
     *                  resized to N × Q (col-major) and overwritten.
     * @param eval      Evaluator class containing the scratch buffers for evaluation.
     */
    virtual void eval_on_span(const Vector<T>& points, Index span_idx,
                              std::vector<Matrix<T>>& results,
                              Evaluator<T>& eval) const = 0;

    // === Utility Methods ============================================================

    /**
     * @brief Find the knot span containing the given parameter value.
     *
     * Convenience wrapper for knot_vector().find_span(degree(), param).
     *
     * @param param Parameter value.
     * @return Span index.
     */
    Index find_span(T param) const { return knots_.find_span(degree_, param); }

    /**
     * @brief Compute the Greville abscissae for this basis.
     *
     * @details The Greville abscissa of basis function `N_{i,p}` is the
     *          average of its `p` interior knots,
     *          `xi_i = (1/p) Σ_{j=1..p} knots[i+j]`. They satisfy linear
     *          precision and serve as natural node points for interpolation
     *          and collocation. Independent of weights — same formula for
     *          B-splines and NURBS, hence implemented here on the base class.
     *
     * @return A vector of parameter values corresponding to the Greville points.
     */
    Vector<T> greville_abscissae() const
    {
        const Index p = degree_;
        const Index n = knots_.size() - p - 1;

        Vector<T> greville(n);
        for (Index i = 0; i < n; ++i) {
            T sum = 0;
            for (Index j = 1; j <= p; ++j) {
                sum += knots_[i + j];
            }
            greville[i] = p > 0 ? sum / static_cast<T>(p) : knots_[i];
        }
        return greville;
    }

    // === Refinement =================================================================

    /**
     * @brief Refine the basis by inserting a single copy of knot value `u`.
     *
     * @param u Knot value to insert. Must lie in [knots.front(), knots.back()].
     * @return A pair of (refined basis, control-point transform matrix).
     *         The transform maps old CPs to new ones.
     */
    virtual std::pair<Ptr<Basis<T>>, Matrix<T>> insert_knot(T u) const = 0;

    /**
     * @brief Refine the basis by elevating the polynomial degree by one.
     *
     * @details Continuity at every existing internal knot is preserved. Each 
     *          unique knot's multiplicity is incremented by one.
     *
     * @return  A pair of (refined basis, control-point transform matrix).
     *         The transform maps old CPs to new ones.
     */
    virtual std::pair<Ptr<Basis<T>>, Matrix<T>> elevate_degree() const = 0;

    // === Properties =================================================================

    /// @brief Get the knot vector object
    const KnotVector<T>& knot_vector() const { return knots_; }

    /// @brief Get the raw knot values
    const std::vector<T>& knots() const { return knots_.data(); }

    /// @brief Get the degree of the basis functions
    Index degree() const { return degree_; }

    /// @brief Get the number of basis functions
    virtual Index num_basis() const { return knots_.size() - degree_ - 1; }

    /// @brief Get the number of parametric intervals (elements) for this basis
    virtual Index num_intervals() const { return knots_.num_spans(); }

protected:

    /// @brief Degree of the basis functions
    Index degree_;

    /// @brief Knot vector defining the basis functions
    KnotVector<T> knots_;
};

} // namespace pyck

#endif // PYCK_BASIS_HPP
