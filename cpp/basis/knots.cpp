#include "knots.hpp"

#include <stdexcept>

namespace pyck
{

KnotVector KnotVector::clamped_uniform(std::size_t degree, std::size_t num_basis)
{
    // Math check: m = n + p + 1
    // Number of knots = num_basis + degree + 1
    if (num_basis < degree + 1) {
        throw std::invalid_argument(
            "Number of basis functions must be at least degree + 1."
        );
    }

    std::size_t num_knots = num_basis + degree + 1;
    std::vector<double> knots(num_knots);

    // Number of unique "spans" in the internal part of the vector
    // For a clamped vector, there are (num_basis - degree) internal segments
    std::size_t num_internal_spans = num_basis - degree;

    for (std::size_t i = 0; i < num_knots; ++i) {
        if (i <= degree) {
            // Clamped start: first p+1 knots are 0.0
            knots[i] = 0.0;
        } 
        else if (i >= num_knots - degree - 1) {
            // Clamped end: last p+1 knots are 1.0
            knots[i] = 1.0;
        } 
        else {
            // Uniformly spaced internal knots
            // Example: p=2, n=4 -> knots = [0, 0, 0, 0.33, 0.66, 1, 1, 1]
            knots[i] = static_cast<double>(i - degree) / num_internal_spans;
        }
    }

    return KnotVector(std::move(knots));
}

std::size_t KnotVector::find_span(std::size_t degree, double u) const
{
    std::size_t num_knots = knots_.size();

    // Basis indexing 0 .. n
    int n = num_knots - degree - 2;
    // Edge cases
    if (u >= knots_[n + 1]) return n;
    if (u <= knots_[degree]) return degree;
    // Binary search for the span
    auto it = std::upper_bound(knots_.begin() + degree, knots_.begin() + n + 1, u);
    return std::distance(knots_.begin(), it) - 1;
}

} // namespace pyck