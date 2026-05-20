#ifndef PYCK_QUADRATURE_HPP
#define PYCK_QUADRATURE_HPP

#include <vector>
#include <concepts>
#include <array>
#include <stdexcept>
#include <Eigen/Dense>

#include "../types.hpp"

namespace pyck
{

/**
 * @brief Base class for quadrature rules on a d-dimensional parametric domain.
 * 
 * Supports tensor-product constructions of multi-dimensional rules from 1D rules,
 * as well as mapping to arbitrary boxes in parametric space.
 * 
 * @tparam T Scalar floating-point type
 * @tparam d Parametric dimension (default = 1)
 */
template <std::floating_point T, std::size_t d = 1>
class QuadratureRule
{
public:

    /// @brief Default constructor.
    QuadratureRule() = default;

    /**
     * @brief Construct a QuadratureRule from a given set of points and weights.
     * 
     * @param pts The quadrature points (reference domain [-1, 1]^d).
     * @param w   The quadrature weights.
     */
    QuadratureRule(const ColMatrix<T, d>& pts, const Vector<T>& w);

    /**
     * @brief Construct an anisotropic d-dimensional rule from d one-dimensional rules.
     * 
     * Forms the tensor product of the provided one-dimensional rules.
     * 
     * @param rules Array of d 1D quadrature rule pointers.
     */
    explicit QuadratureRule(const std::array<const QuadratureRule<T, 1>*, d>& rules);

    /**
     * @brief Construct an isotropic d-dimensional rule from a single 1D rule.
     * 
     * @param rule The 1D rule to be used in all directions.
     */
    explicit QuadratureRule(const QuadratureRule<T, 1>& rule);

    /// @brief Virtual destructor.
    virtual ~QuadratureRule() = default;

    /**
     * @brief Form the tensor product of d one-dimensional quadrature rules.
     * 
     * @param rules Array of d 1D quadrature rule pointers.
     * @return A pair containing (points, weights) of the product rule.
     */
    static std::pair<ColMatrix<T, d>, Vector<T>> tensor_product(const std::array<const QuadratureRule<T, 1>*, d>& rules);

    /**
     * @brief Map the reference [-1, 1]^d quadrature points and weights to a box [lo, hi].
     * 
     * @param lo Lower bounds of the target interval in each dimension.
     * @param hi Upper bounds of the target interval in each dimension.
     * @return A pair (mapped_points, mapped_weights).
     */
    virtual std::pair<ColMatrix<T, d>, Vector<T>> map_to_domain(const std::array<T, d>& lo, 
                                                                const std::array<T, d>& hi) const;

    /// @brief Get the quadrature points as a matrix of size Q x d.
    const ColMatrix<T, d>& points() const { return points_; }

    /// @brief Get the quadrature weights as a vector of length Q.
    const Vector<T>& weights() const { return weights_; }

    /// @brief Get the number of quadrature points.
    std::size_t num_points() const { return weights_.size(); }

    /// @brief Get the parametric dimension.
    static constexpr std::size_t dim() { return d; }

protected:

    /// @brief Quadrature points in reference coordinates, matrix of size Q x d.
    ColMatrix<T, d> points_;

    /// @brief Quadrature weights, vector of length Q.
    Vector<T> weights_;
};

/**
 * @brief Partial specialization for one-dimensional quadrature rules.
 * 
 * Introduces virtual methods for computation and lookup used by specific
 * implementations like GaussLegendre.
 */
template <std::floating_point T>
class QuadratureRule<T, 1>
{
public:
    /// @brief Default constructor.
    QuadratureRule() = default;

    /**
     * @brief Construct a 1D QuadratureRule from points and weights.
     * 
     * @param pts The quadrature points (reference domain [-1, 1]).
     * @param w   The quadrature weights.
     */
    QuadratureRule(const ColMatrix<T, 1>& pts, const Vector<T>& w);

    /// @brief Virtual destructor.
    virtual ~QuadratureRule() = default;

    /**
     * @brief Map the reference [-1, 1] quadrature points and weights to an interval [lo, hi].
     * 
     * @param lo Lower bound of the target interval.
     * @param hi Upper bound of the target interval.
     * @return A pair (mapped_points, mapped_weights).
     */
    virtual std::pair<ColMatrix<T, 1>, Vector<T>> map_to_domain(
        const std::array<T, 1>& lo, 
        const std::array<T, 1>& hi) const;

    /**
     * @brief 1D mapping helper taking scalars.
     * 
     * @param lo Lower bound of the target interval.
     * @param hi Upper bound of the target interval.
     * @return A pair (mapped_points_vector, mapped_weights_vector).
     */
    std::pair<Vector<T>, Vector<T>> map_to_domain(T lo, T hi) const;

    /**
     * @brief Form the tensor product of 1 one-dimensional quadrature rule (identity).
     * 
     * @param rules Array containing a single 1D quadrature rule pointer.
     * @return A pair containing (points, weights) of the rule.
     */
    static std::pair<ColMatrix<T, 1>, Vector<T>> tensor_product(
        const std::array<const QuadratureRule<T, 1>*, 1>& rules);

    /// @brief Get the quadrature points as a matrix rows=Q, cols=1.
    const ColMatrix<T, 1>& points() const 
    { return points_; }

    /// @brief Get the quadrature weights as a vector of length Q.
    const Vector<T>& weights() const 
    { return weights_; }

    /// @brief Get the number of quadrature points.
    std::size_t num_points() const 
    { return weights_.size(); }

    /// @brief Get the parametric dimension.
    static constexpr std::size_t dim() 
    { return 1; }

protected:
    /**
     * @brief Compute the reference nodes and weights if not found in lookup.
     * @param num_pts Target number of points.
     * @param nodes   Reference nodes (output).
     * @param weights Reference weights (output).
     */
    virtual void compute_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const {}
    
    /**
     * @brief Search for precomputed reference nodes and weights.
     * @param num_pts Target number of points.
     * @param nodes   Reference nodes (output).
     * @param weights Reference weights (output).
     * @return True if found, false otherwise.
     */
    virtual bool lookup_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const { return false; }

    /// @brief Quadrature points
    ColMatrix<T, 1> points_;

    /// @brief Quadrature weights
    Vector<T> weights_;
};

} // namespace pyck

#endif // PYCK_QUADRATURE_HPP
