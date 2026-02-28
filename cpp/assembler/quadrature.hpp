#ifndef PYCK_QUADRATURE_HPP
#define PYCK_QUADRATURE_HPP

#include <vector>
#include <Eigen/Dense>

namespace pyck
{

class QuadratureRule
{
public:

    // === Constructors ===============================================================

    QuadratureRule() = default;

    // === Properties =================================================================

    /// @brief Get the quadrature points as a (Q, dim) matrix.
    const Eigen::MatrixXd& points() const { return points_; }

    /// @brief Get the quadrature weights as a vector.
    const Eigen::VectorXd& weights() const { return weights_; }

    /// @brief Get the number of quadrature points.
    std::size_t num_points() const { return weights_.size(); }

    /// @brief Get the dimension of the quadrature points.
    std::size_t dim() const { return points_.cols(); }

    // === Methods ====================================================================

    /**
     * @brief Construct a new QuadratureRule that is the tensor product of this rule and another rule.
     * 
     * The resulting quadrature rule will have points that are the Cartesian product of the points of the two rules,
     * and weights that are the product of the corresponding weights.
     */
    QuadratureRule tensor_product(const QuadratureRule& other) const;

protected:
    /// @brief Quadrature points stored as a (Q, dim) matrix.
    Eigen::MatrixXd points_;

    /// @brief Quadrature weights stored as a vector of length Q.
    Eigen::VectorXd weights_;
};

} // namespace pyck

#endif // PYCK_QUADRATURE_HPP