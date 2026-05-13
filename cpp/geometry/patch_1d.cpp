#include "patch.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u, const ColMatrix<T, 3>& control_pts) requires(d == 1)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u)),
      dof_mapper_({tensor_product_.basis(0).num_basis()}, {tensor_product_.basis(0).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Control points must be embedded in 3D space.");
    }
    const Index expected_n = this->tensor_product_.basis(0).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());
    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch<T, 1>: "
                                    "Dimension mismatch.");
    }

    // Initialize Greville points and spans
    Vector<T> gu = this->tensor_product_.basis(0).greville_abscissae();
    greville_points_.resize(gu.size(), 1);
    greville_spans_.resize(gu.size());
    for (Index i = 0; i < static_cast<Index>(gu.size()); ++i) {
        greville_points_(i, 0) = gu[i];
        greville_spans_[i] = this->tensor_product_.basis(0).find_span(gu[i]);
    }
}

template class Patch<double, 1>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 1>;
#endif

} // namespace pyck
