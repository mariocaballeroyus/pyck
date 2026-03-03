#ifndef PYCK_QUADRATURE_HPP
#define PYCK_QUADRATURE_HPP

#include <vector>
#include <concepts>
#include <array>
#include <Eigen/Dense>

#include "../types.hpp"

namespace pyck
{

/// @brief Abstract base class for strictly one-dimensional quadrature rules.
/// @tparam T Scalar type
template <std::floating_point T>
class QuadratureRule
{
public:

    /// @brief Default constructor.
    QuadratureRule() = default;

    /**
     * @brief Construct a QuadratureRule from a given set of 1D points and weights.
     * 
     * @param pts The 1D quadrature points (reference domain [-1, 1]).
     * @param w   The quadrature weights.
     */
    QuadratureRule(const Vector<T>& pts, const Vector<T>& w)
        : points_(pts), weights_(w) {}

    virtual ~QuadratureRule() = default;

    /**
     * @brief Map the reference [-1, 1] quadrature points and weights to an interval [lo, hi].
     * 
     * @param lo Lower bound of the target interval.
     * @param hi Upper bound of the target interval.
     * @return A pair (mapped_points, mapped_weights).
     */
    std::pair<Vector<T>, Vector<T>> map_to_domain(T lo, T hi) const;

    
    /// @brief Get the 1D quadrature points as a vector of length Q.
    const Vector<T>& points() const { return points_; }

    /// @brief Get the quadrature weights as a vector of length Q.
    const Vector<T>& weights() const { return weights_; }

    /// @brief Get the number of quadrature points.
    std::size_t num_points() const { return weights_.size(); }

protected:

    /**
     * @brief Compute the 1D reference nodes and weights for a given number of points.
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes   Output quadrature nodes on [-1, 1].
     * @param weights Output quadrature weights.
     */
    virtual void compute_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const = 0;

    /**
     * @brief Look up precomputed 1D reference nodes and weights.
     * 
     * @param num_pts The number of quadrature points.
     * @param nodes   Output quadrature nodes on [-1, 1].
     * @param weights Output quadrature weights.
     * @return true if the lookup succeeded, false otherwise.
     */
    virtual bool lookup_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const = 0;

    /// @brief 1D quadrature points, vector of length Q.
    Vector<T> points_;

    /// @brief 1D quadrature weights, vector of length Q.
    Vector<T> weights_;

};

// === Tensor-Product Free Functions ==================================================

/**
 * @brief Form the tensor product of d one-dimensional quadrature rules.
 *
 * @tparam T  Scalar type.
 * @tparam d  Number of parametric dimensions.
 * @param rules  Array of pointers to 1D quadrature rules (one per dimension).
 * @return A pair (points, weights) where points is a (Q_total, d) matrix and
 *         weights is a vector of length Q_total.
 */
template <std::floating_point T, std::size_t d>
std::pair<ColMatrix<T, d>, Vector<T>> tensor_product(
    const std::array<const QuadratureRule<T>*, d>& rules)
{
    std::size_t total_q = 1;
    for (std::size_t i = 0; i < d; ++i) {
        total_q *= rules[i]->num_points();
    }

    ColMatrix<T, d> points(total_q, d);
    Vector<T> weights(total_q);

    for (std::size_t i = 0; i < total_q; ++i)
    {
        std::size_t temp = i;
        T w = static_cast<T>(1);
        for (std::size_t j = d; j-- > 0; )
        {
            std::size_t qj = rules[j]->num_points();
            std::size_t idx = temp % qj;
            temp /= qj;

            points(i, j) = rules[j]->points()(idx);
            w *= rules[j]->weights()(idx);
        }
        weights(i) = w;
    }
    return {points, weights};
}

/**
 * @brief Form the tensor product of d one-dimensional quadrature rules and
 *        map the result from [-1,1]^d to the box [lo, hi].
 *
 * @tparam T  Scalar type.
 * @tparam d  Number of parametric dimensions.
 * @param rules        Array of pointers to 1D quadrature rules.
 * @param lower_bounds Lower corner of the target domain.
 * @param upper_bounds Upper corner of the target domain.
 * @return A pair (mapped_points, mapped_weights).
 */
template <std::floating_point T, std::size_t d>
std::pair<ColMatrix<T, d>, Vector<T>> tensor_product_mapped(
    const std::array<const QuadratureRule<T>*, d>& rules,
    const std::array<T, d>& lower_bounds,
    const std::array<T, d>& upper_bounds)
{
    auto [pts, wts] = tensor_product<T, d>(rules);
    std::size_t Q = pts.rows();
    for (std::size_t q = 0; q < Q; ++q)
    {
        for (std::size_t i = 0; i < d; ++i)
        {
            T a = lower_bounds[i];
            T b = upper_bounds[i];
            T xi = pts(q, i);
            pts(q, i) = a + static_cast<T>(0.5) * (b - a) * (xi + static_cast<T>(1));
            wts(q) *= static_cast<T>(0.5) * (b - a);
        }
    }
    return {pts, wts};
}

} // namespace pyck

#endif // PYCK_QUADRATURE_HPP
