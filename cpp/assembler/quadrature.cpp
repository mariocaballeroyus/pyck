#include "quadrature.hpp"

namespace pyck
{

QuadratureRule QuadratureRule::tensor_product(const QuadratureRule& other) const
{
    QuadratureRule result;
    std::size_t q1 = this->num_points();
    std::size_t q2 = other.num_points();
    std::size_t dim1 = this->dim();
    std::size_t dim2 = other.dim();

    result.points_.resize(q1 * q2, dim1 + dim2);
    result.weights_.resize(q1 * q2);

    for (std::size_t i = 0; i < q1; ++i)
    {
        for (std::size_t j = 0; j < q2; ++j)
        {
            std::size_t idx = i * q2 + j;

            // Concatenate the coordinates
            result.points_.row(idx).head(dim1) = this->points().row(i);
            result.points_.row(idx).tail(dim2) = other.points().row(j);

            // Multiply the weights
            result.weights_(idx) = this->weights()(i) * other.weights()(j);
        }
    }
    return result;
}

} // namespace pyck