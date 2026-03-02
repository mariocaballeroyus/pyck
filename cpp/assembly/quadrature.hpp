#ifndef PYCK_QUADRATURE_HPP
#define PYCK_QUADRATURE_HPP

#include <vector>
#include <concepts>
#include <array>
#include <Eigen/Dense>

#include "../types.hpp"

namespace pyck
{

/// @brief Abstract base class for quadrature rules.
/// @tparam T Scalar type
/// @tparam d Dimension
template <std::floating_point T, std::size_t d>
class QuadratureRule
{
public:

    // === Constructors ===============================================================

    /// @brief Default constructor.
    QuadratureRule() = default;

    /**
     * @brief Construct a QuadratureRule from a given set of points and weights.
     * 
     * @param pts The quadrature points.
     * @param w The quadrature weights.
     */
    QuadratureRule(const ColMatrix<T, d>& pts, const Vector<T>& w)
        : points_(pts), weights_(w) {}

    /**
     * @brief Construct an isotropic QuadratureRule.
     * 
     * @param num_pts The number of points per dimension uniformly.
     */
    /**
     * @brief Construct an isotropic QuadratureRule.
     * 
     * @param num_pts The number of points per dimension uniformly.
     */
    explicit QuadratureRule(std::size_t num_pts)
    {
        std::array<std::size_t, d> arr;
        arr.fill(num_pts);
        this->build_tensor_product(arr);
    }
    
    /**
     * @brief Construct an anisotropic QuadratureRule.
     * 
     * @param num_pts An array indicating points per dimension independently.
     */
    explicit QuadratureRule(const std::array<std::size_t, d>& num_pts) {}

    virtual ~QuadratureRule() = default;

    // === Properties =================================================================

    /// @brief Get the quadrature points as a (Q, dim) matrix.
    const ColMatrix<T, d>& points() const { return points_; }

    /// @brief Get the quadrature weights as a vector.
    const Vector<T>& weights() const { return weights_; }

    /// @brief Get the number of quadrature points.
    std::size_t num_points() const { return weights_.size(); }

    /// @brief Get the dimension of the quadrature points.
    static constexpr std::size_t dim() { return d; }

    /**
     * @brief Map the reference [-1, 1]^d quadrature points and weights to a specific tensor-product domain.
     * 
     * @param lower_bounds The lower bounds of the domain in each dimension.
     * @param upper_bounds The upper bounds of the domain in each dimension.
     * @return A pair containing the mapped points matrix and the mapped weights vector.
     */
    std::pair<ColMatrix<T, d>, Vector<T>> map_to_domain(
        const std::array<T, d>& lower_bounds,
        const std::array<T, d>& upper_bounds) const;

protected:

    /**
     * @brief Compute the 1D reference nodes and weights for a given number of points.
     * 
     * @param num_pts The number of points per dimension uniformly.
     * @param nodes The quadrature points.
     * @param weights The quadrature weights.
     */
    virtual void compute_reference(std::size_t num_pts, ColMatrix<T, 1>& nodes, Vector<T>& weights) const = 0;

    /**
     * @brief Lookup the 1D reference nodes and weights for a given number of points.
     * 
     * @param num_pts The number of points per dimension uniformly.
     * @param nodes The quadrature points.
     * @param weights The quadrature weights.
     */
    virtual bool lookup_reference(std::size_t num_pts, ColMatrix<T, 1>& nodes, Vector<T>& weights) const = 0;

    /**
     * @brief Build the tensor product explicitly calling virtual methods.
     * 
     * @param num_pts An array indicating points per dimension independently.
     */
    void build_tensor_product(const std::array<std::size_t, d>& num_pts);

    // === Member Variables ===========================================================

    /// @brief Quadrature points stored as a (Q, dim) matrix.
    ColMatrix<T, d> points_;

    /// @brief Quadrature weights stored as a vector of length Q.
    Vector<T> weights_;

};



} // namespace pyck

#endif // PYCK_QUADRATURE_HPP
