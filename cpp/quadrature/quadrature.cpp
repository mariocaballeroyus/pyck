#include "quadrature.hpp"
#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
QuadratureRule<T, d>::QuadratureRule(const ColMatrix<T, d>& pts, const Vector<T>& w)
    : points_(pts), weights_(w) {}

template <std::floating_point T, std::size_t d>
QuadratureRule<T, d>::QuadratureRule(const std::array<const QuadratureRule<T, 1>*, d>& rules)
{
    auto [pts, wts] = QuadratureRule<T, d>::tensor_product(rules);
    this->points_ = std::move(pts);
    this->weights_ = std::move(wts);
}

template <std::floating_point T, std::size_t d>
QuadratureRule<T, d>::QuadratureRule(const QuadratureRule<T, 1>& rule)
{
    std::array<const QuadratureRule<T, 1>*, d> rules;
    rules.fill(&rule);
    auto [pts, wts] = QuadratureRule<T, d>::tensor_product(rules);
    this->points_ = std::move(pts);
    this->weights_ = std::move(wts);
}

template <std::floating_point T, std::size_t d>
void
QuadratureRule<T, d>::map_to_domain(const std::array<T, d>& lo,
                                    const std::array<T, d>& hi,
                                    Eigen::Ref<ColMatrix<T, d>, 0, Eigen::OuterStride<>> pts_out,
                                    Eigen::Ref<Vector<T>> wts_out) const
{
    T det_j_total = static_cast<T>(1);
    for (std::size_t i = 0; i < d; ++i) {
        const T det_j = static_cast<T>(0.5) * (hi[i] - lo[i]);
        pts_out.col(i).array() = lo[i] + det_j * (points_.col(i).array() + static_cast<T>(1));
        det_j_total *= det_j;
    }
    wts_out = weights_ * det_j_total;
}

template <std::floating_point T, std::size_t d>
std::pair<ColMatrix<T, d>, Vector<T>> QuadratureRule<T, d>::tensor_product(
    const std::array<const QuadratureRule<T, 1>*, d>& rules)
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
        for (std::size_t j = 0; j < d; ++j) 
        {
            std::size_t qj = rules[j]->num_points();
            std::size_t idx = temp % qj;
            temp /= qj;

            points(i, j) = rules[j]->points()(idx, 0); 
            w *= rules[j]->weights()(idx);
        }
        weights(i) = w;
    }
    return {points, weights};
}

template <std::floating_point T>
QuadratureRule<T, 1>::QuadratureRule(const ColMatrix<T, 1>& pts, const Vector<T>& w)
    : points_(pts), weights_(w) {}

template <std::floating_point T>
void
QuadratureRule<T, 1>::map_to_domain(const std::array<T, 1>& lo,
                                    const std::array<T, 1>& hi,
                                    Eigen::Ref<ColMatrix<T, 1>, 0, Eigen::OuterStride<>> pts_out,
                                    Eigen::Ref<Vector<T>> wts_out) const
{
    const T det_j = static_cast<T>(0.5) * (hi[0] - lo[0]);
    pts_out.col(0).array() = lo[0] + det_j * (points_.col(0).array() + static_cast<T>(1));
    wts_out = weights_ * det_j;
}

template <std::floating_point T>
std::pair<ColMatrix<T, 1>, Vector<T>>
QuadratureRule<T, 1>::tensor_product(const std::array<const QuadratureRule<T, 1>*, 1>& rules)
{
    return {rules[0]->points(), rules[0]->weights()};
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
