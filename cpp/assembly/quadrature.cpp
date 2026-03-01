#include "quadrature.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
std::pair<ColMatrix<T, d>, Vector<T>> QuadratureRule<T, d>::map_to_domain(
    const std::array<T, d>& lower_bounds,
    const std::array<T, d>& upper_bounds) const
{
    std::size_t Q = num_points();
    ColMatrix<T, d> mapped_pts(Q, d);
    Vector<T> mapped_weights = weights_;
    
    for (std::size_t q = 0; q < Q; ++q) 
    {
        for (std::size_t i = 0; i < d; ++i) 
        {
            T a = lower_bounds[i];
            T b = upper_bounds[i];
            T xi = points_(q, i);
            
            mapped_pts(q, i) = a + static_cast<T>(0.5) * (b - a) * (xi + static_cast<T>(1.0));
            mapped_weights(q) *= static_cast<T>(0.5) * (b - a);
        }
    }
    return {mapped_pts, mapped_weights};
}

template <std::floating_point T, std::size_t d>
void QuadratureRule<T, d>::build_tensor_product(const std::array<std::size_t, d>& num_pts)
{
    std::size_t total_q = 1;
    for (std::size_t i = 0; i < d; ++i) {
        total_q *= num_pts[i];
    }

    this->points_.resize(total_q, Eigen::NoChange);
    this->weights_.resize(total_q);

    std::array<ColMatrix<T, 1>, d> all_nodes;
    std::array<Vector<T>, d> all_weights;

    for (std::size_t i = 0; i < d; ++i) {
        if (!this->lookup_reference(num_pts[i], all_nodes[i], all_weights[i])) {
            this->compute_reference(num_pts[i], all_nodes[i], all_weights[i]);
        }
    }

    for (std::size_t i = 0; i < total_q; ++i)
    {
        std::size_t temp = i;
        T weight = 1.0;
        // Lexicographical ordering: last dimension varies fastest
        for (std::size_t j = d; j-- > 0; )
        {
            std::size_t qj = num_pts[j];
            std::size_t idx = temp % qj;
            temp /= qj;

            this->points_(i, j) = all_nodes[j](idx, 0);
            weight *= all_weights[j](idx);
        }
        this->weights_(i) = weight;
    }
}

// === Template Instantiations ========================================================

template class pyck::QuadratureRule<double, 1>;
template class pyck::QuadratureRule<double, 2>;
template class pyck::QuadratureRule<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class pyck::QuadratureRule<float, 1>;
template class pyck::QuadratureRule<float, 2>;
template class pyck::QuadratureRule<float, 3>;
#endif

} // namespace pyck
