#ifndef PYCK_BASIS_HPP
#define PYCK_BASIS_HPP

#include <cstddef>
#include <vector>
#include <concepts>

#include "bspline_algorithms.hpp"
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
     * @param points      Parameter values (all assumed to lie in the same knot span).
     * @param span_idx    Knot-span index (as returned by KnotVector::find_span).
     * @param order       Maximum derivative order.
     * @param uni_results Output buffer, pre-sized to `order + 1`. Each entry is
     *                    resized to N × Q (col-major) and overwritten.
     */
    virtual void eval_on_span(const Vector<T>& points, Index span_idx, Index order,
                              std::vector<Matrix<T>>& uni_results) const = 0;

    /**
     * @brief Evaluate basis functions and derivatives up to @p order at @p points,
     *        scattering the active span-local values into the global (m × n) layout.
     *
     * @param points Parameter values; each may lie in a different knot span.
     * @param order  Maximum derivative order.
     * @return Per-order matrices, each sized (points.size() × num_basis()),
     *         with zeros outside the active (p+1) functions for each point.
     */
    std::vector<Matrix<T>> eval_all(const Vector<T>& points, Index order) const
    {
        const Index m = static_cast<Index>(points.size());
        const Index n = num_basis();
        const Index p = degree_;

        std::vector<Matrix<T>> results(order + 1);
        for (Index k = 0; k <= order; ++k)
            results[k] = Matrix<T>::Zero(m, n);

        std::vector<Matrix<T>> local(order + 1);
        Vector<T> pt(1);
        for (Index i = 0; i < m; ++i) {
            const Index span = find_span(points[i]);
            pt[0] = points[i];
            eval_on_span(pt, span, order, local);
            for (Index k = 0; k <= order; ++k) {
                results[k].row(i).segment(span - p, p + 1) = local[k].col(0).transpose();
            }
        }
        return results;
    }

    // === Utility Methods ============================================================

    /**
     * @brief Find the knot span containing the given parameter value.
     *
     * @param param Parameter value.
     * @return Span index.
     */
    Index find_span(T param) const { return knots_.find_span(degree_, param); }

    /**
     * @brief Compute the Greville abscissae for this basis.
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

    /// @brief Polymorphic deep copy (used by basis-level refinement helpers).
    virtual Ptr<Basis<T>> clone() const = 0;

    /**
     * @brief C⁰ Bezier extraction: refine until every interior knot reaches
     *        multiplicity `degree`, so the basis splits into independent
     *        Bernstein segments — one per non-empty knot span.
     *
     * @details This is the standard operator behind per-element Bezier
     *          representations (e.g. VTK Bezier cells). It is rational-aware:
     *          on a NURBS basis the returned basis carries the extracted
     *          weights and the transform is the rational (Euclidean→Euclidean)
     *          operator, so each segment is a rational Bezier.
     *
     * @return A pair of (extracted basis, transform `C`). `C` is `(n_ext × n)`
     *         and maps this basis's control values to the extracted ones:
     *         `ext = C · old`.
     */
    std::pair<Ptr<Basis<T>>, Matrix<T>> bezier_extract() const
    {
        const Index p = degree_;
        Ptr<Basis<T>> current = clone();
        Matrix<T> C = Matrix<T>::Identity(num_basis(), num_basis());

        const T lo = knots_.front();
        const T hi = knots_.back();
        for (Index i = 0; i < knots_.size(); ++i) {
            const T u = knots_[i];
            if (u <= lo || u >= hi)        continue;  // boundary knot
            if (i > 0 && knots_[i - 1] == u) continue;  // visit each value once

            Index mult = 0;
            const KnotVector<T>& kv = current->knot_vector();
            for (Index j = 0; j < kv.size(); ++j)
                if (kv[j] == u) ++mult;

            for (Index r = mult; r < p; ++r) {
                auto [refined, step] = current->insert_knot(u);
                C = step * C;
                current = std::move(refined);
            }
        }
        return {std::move(current), std::move(C)};
    }

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
