#include "beam_euler_bernoulli_1p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

template <std::floating_point T>
BeamEulerBernoulli1p<T>::BeamEulerBernoulli1p(Ptr<UniaxialStress1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamEulerBernoulli1p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
Matrix<T>
BeamEulerBernoulli1p<T>::strain_matrix(const Patch<T, 1>& /*patch*/,
                                       const BasisValues<T, 1>& basis,
                                       const IntrinsicGeometry<T, 1>& ig) const
{
    const Index Q = basis.Q();
    const Index N = basis.N();
    Matrix<T> B(Q, N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = basis.data()[1].col(q);  // N
        auto slab2 = basis.data()[2].col(q);  // N

        const T Gam = ig.chr.Gamma(0, 0, 0)(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_u_i  = slab1(i);
            const T N_uu_i = slab2(i);

            // Bending: B_i = -N_{i|11}
            B(q, i) = -(N_uu_i - Gam * N_u_i);
            // B_s = 0 (normality assumption)
        }
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
                                                   const BasisValues<T, 1>& basis,
                                                   const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_w = [ N_i ]
    const Index Q = basis.Q();
    const Index N = basis.N();
    Matrix<T> N_w(Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = basis.data()[0].col(q);
        for (Index i = 0; i < N; ++i) {
            N_w(q, i) = slab0(i);
        }
    }
    return N_w;
}

template <std::floating_point T>
Matrix<T>
BeamEulerBernoulli1p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                               const BasisValues<T, 1>& basis,
                                               const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_rot = [ -N_{i|1} ]
    const Index Q = basis.Q();
    const Index N = basis.N();
    Matrix<T> N_varphi(Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = basis.data()[1].col(q);
        for (Index i = 0; i < N; ++i) {
            N_varphi(q, i) = -slab1(i);
        }
    }
    return N_varphi;
}

// === Template Instantiations ========================================================

template class BeamEulerBernoulli1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamEulerBernoulli1p<float>;
#endif

} // namespace pyck
