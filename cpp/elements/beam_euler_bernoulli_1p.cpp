#include "beam_euler_bernoulli_1p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
BeamEulerBernoulli1p<T>::BeamEulerBernoulli1p(Ptr<UniaxialStress1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamEulerBernoulli1p: "
                                    "material is null.");
    }
}

// === Matrix Operators =============================================================??

template <std::floating_point T>
Matrix<T>
BeamEulerBernoulli1p<T>::strain_matrix(const Patch<T, 1>& /*patch*/,
                                       const BasisDerivs<T, 1>& basis,
                                       const IntrinsicGeometry<T, 1>& ig) const
{
    const Index Q = basis.N_d1(0).rows();
    const Index n = basis.N_d1(0).cols();
    Matrix<T> B(Q, n);


    for (Index q = 0; q < Q; ++q)
    {
        // Bending
        // B_b = [ -N_{i|11} ]
        B.row(q) = -(basis.N_d2(0, 0).row(q) - ig.chr.Gamma(0, 0, 0)(q) * basis.N_d1(0).row(q));

        // B_s = 0 (normality assumption)
    }
    return B;
}

template <std::floating_point T>
Matrix<T> 
BeamEulerBernoulli1p<T>::constitutive_matrix(const IntrinsicGeometry<T, 1>& ig,
                                             Index q) const
{
    const T gi = ig.g_inv(0, 0)(q);
    Matrix<T> D(1, 1);

    // D_b = [ EI (g^{11})^2 ]
    D(0, 0) = material_->bending_stiffness() * gi * gi;
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
BeamEulerBernoulli1p<T>::displacement_shape_matrix(const Patch<T, 1>& /*patch*/,
                                                   const BasisDerivs<T, 1>& basis,
                                                   const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_w = [ N_i ]
    return basis.N();
}

template <std::floating_point T>
Matrix<T>
BeamEulerBernoulli1p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                               const BasisDerivs<T, 1>& basis,
                                               const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_rot = [ -N_{i|1} ]
    return -basis.N_d1(0);
}

// === Template Instantiations ========================================================

template class BeamEulerBernoulli1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamEulerBernoulli1p<float>;
#endif

} // namespace pyck
