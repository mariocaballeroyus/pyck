#include "beam_timoshenko_1p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

template <std::floating_point T>
BeamTimoshenko1p<T>::BeamTimoshenko1p(Ptr<UniaxialStress1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamTimoshenko1p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
Matrix<T>
BeamTimoshenko1p<T>::strain_matrix(const Patch<T, 1>& /*patch*/,
                                   const std::vector<Matrix<T>>& basis,
                                   const IntrinsicGeometry<T, 1>& ig) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> B(2 * Q, N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = basis[1].col(q);
        auto slab2 = basis[2].col(q);
        auto slab3 = basis[3].col(q);

        const T gi          = ig.g_inv(0, 0)(q);
        const T G           = ig.chr.Gamma(0, 0, 0)(q);
        const T G2          = G * G;
        const T a11_dot_a11 = ig.a_d1(0, 0).row(q).squaredNorm();
        const T a1_dot_a111 = ig.a(0).row(q).dot(ig.a_d2(0, 0, 0).row(q));
        const T coeff_Nu    = T(4) * G2 - gi * (a11_dot_a11 + a1_dot_a111);
        const T coeff_Nuu   = -T(3) * G;

        for (Index i = 0; i < N; ++i)
        {
            const T N_u_i   = slab1(i);
            const T N_uu_i  = slab2(i);
            const T N_uuu_i = slab3(i);

            // Bending: B_b = -N_{i|11}
            B(2*q,     i) = -(N_uu_i - G * N_u_i);
            // Transverse Shear: B_s = -(K_b/K_s) g^{11} N_{i|111}
            B(2*q + 1, i) = -ratio * gi
                          * (N_uuu_i + coeff_Nuu * N_uu_i + coeff_Nu * N_u_i);
        }
    }
    return B;
}

template <std::floating_point T>
Matrix<T>
BeamTimoshenko1p<T>::constitutive_matrix(const IntrinsicGeometry<T, 1>& ig,
                                         Index q) const
{
    const T gi = ig.g_inv(0, 0)(q);
    Matrix<T> D = Matrix<T>::Zero(2, 2);
    D(0, 0) = material_->bending_stiffness() * gi * gi;
    D(1, 1) = material_->shear_stiffness()   * gi;
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
BeamTimoshenko1p<T>::displacement_shape_matrix(const Patch<T, 1>& /*patch*/,
                                               const std::vector<Matrix<T>>& basis,
                                               const IntrinsicGeometry<T, 1>& ig) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> Nw(Q, N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab0 = basis[0].col(q);
        auto slab1 = basis[1].col(q);
        auto slab2 = basis[2].col(q);

        const T gi  = ig.g_inv(0, 0)(q);
        const T Gam = ig.chr.Gamma(0, 0, 0)(q);
        const T coef = -ratio * gi;

        for (Index i = 0; i < N; ++i)
        {
            const T N_i    = slab0(i);
            const T N_u_i  = slab1(i);
            const T N_uu_i = slab2(i);

            // N_w = N_i - (K_b/K_s) g^{11} (N_{i|11})
            Nw(q, i) = N_i + coef * (N_uu_i - Gam * N_u_i);
        }
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T>
BeamTimoshenko1p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                           const std::vector<Matrix<T>>& basis,
                                           const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_rot = [ -N_{i|1} ]
    const Index Q = basis[0].cols();
    const Index N = basis[0].rows();
    Matrix<T> N_varphi(Q, N);
    for (Index q = 0; q < Q; ++q) {
        auto slab1 = basis[1].col(q);
        for (Index i = 0; i < N; ++i) {
            N_varphi(q, i) = -slab1(i);
        }
    }
    return N_varphi;
}

// === Template Instantiations ========================================================

template class BeamTimoshenko1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko1p<float>;
#endif

} // namespace pyck
