#include "patch.hpp"

#include <stdexcept>

namespace pyck
{

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u, Ptr<const Basis<T>> basis_v, const ColMatrix<T, 3>& control_pts) requires(d == 2)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u), std::move(basis_v)),
      dof_mapper_({tensor_product_.basis(0).num_basis(), tensor_product_.basis(1).num_basis()},
                  {tensor_product_.basis(0).degree(),    tensor_product_.basis(1).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("Patch<T, 2>: Control points must be embedded in 3D space.");
    }
    const Index expected_n = this->tensor_product_.basis(0).num_basis() * this->tensor_product_.basis(1).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());
    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch<T, 2>: Dimension mismatch.");
    }

    // Initialize Greville points and spans
    Vector<T> gu = this->tensor_product_.basis(0).greville_abscissae();
    Vector<T> gv = this->tensor_product_.basis(1).greville_abscissae();
    Index nu = gu.size();
    Index nv = gv.size();

    greville_points_.resize(nu * nv, 2);
    greville_spans_.resize(nu * nv);

    auto intervals = this->tensor_product_.num_intervals();
    for (Index i = 0; i < nu; ++i) {
        Index su = this->tensor_product_.basis(0).find_span(gu[i]);
        for (Index j = 0; j < nv; ++j) {
            Index flat = i * nv + j;
            greville_points_(flat, 0) = gu[i];
            greville_points_(flat, 1) = gv[j];
            Index sv = this->tensor_product_.basis(1).find_span(gv[j]);
            greville_spans_[flat] = su * intervals[1] + sv;
        }
    }
}

template class Patch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 2>;
#endif

} // namespace pyck
