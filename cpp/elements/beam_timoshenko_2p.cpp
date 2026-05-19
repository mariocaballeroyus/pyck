#include "beam_timoshenko_2p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

template <std::floating_point T>
BeamTimoshenko2p<T>::BeamTimoshenko2p(Ptr<UniaxialStress1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamTimoshenko2p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
Matrix<T>
BeamTimoshenko2p<T>::constitutive_matrix(const IntrinsicGeometry<T, 1>& ig,
                                         Index q) const
{
    // D = [ D_b  0   ]
    //     [ 0    D_s ]
    const T gi = ig.g_inv(0, 0)(q);
    Matrix<T> D = Matrix<T>::Zero(2, 2);
    D(0, 0) = material_->bending_stiffness() * gi;
    D(1, 1) = material_->shear_stiffness()   * gi;
    return D;
}

template <std::floating_point T>
Matrix<T>
BeamTimoshenko2p<T>::strain_matrix(const Patch<T, 1>& /*patch*/,
                                   const std::vector<Matrix<T>>& basis,
                                   const IntrinsicGeometry<T, 1>& ig) const
{
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> B = Matrix<T>::Zero(2 * Q, 2 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = basis[0].col(q);
        auto slab1 = basis[1].col(q);

        const T G = ig.chr.Gamma(0, 0, 0)(q);

        for (Index i = 0; i < N; ++i)
        {
            const T N_i   = slab0(i);
            const T N_u_i = slab1(i);

            // Bending: B_b = [ 0    N_{i|1} − Γ^1_{11} N_i ]
            B(2*q,     2*i + 1) = N_u_i - G * N_i;
            // Transverse Shear: B_s = [ N_{i|1}  N_i ]
            B(2*q + 1, 2*i    ) = N_u_i;
            B(2*q + 1, 2*i + 1) = N_i;
        }
    }
    return B;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
BeamTimoshenko2p<T>::displacement_shape_matrix(const Patch<T, 1>& /*patch*/,
                                               const std::vector<Matrix<T>>& basis,
                                               const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_w = [ N_i  0 ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 2 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = basis[0].col(q);
        for (Index i = 0; i < N; ++i) {
            Nw(q, 2*i) = slab0(i);
        }
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T>
BeamTimoshenko2p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                           const std::vector<Matrix<T>>& basis,
                                           const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_rot = [ 0  N_i ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> Nth = Matrix<T>::Zero(Q, 2 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = basis[0].col(q);
        for (Index i = 0; i < N; ++i) {
            Nth(q, 2*i + 1) = slab0(i);
        }
    }
    return Nth;
}

// === Template Instantiations ========================================================

template class BeamTimoshenko2p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko2p<float>;
#endif

} // namespace pyck
