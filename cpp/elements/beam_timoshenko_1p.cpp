#include "beam_timoshenko_1p.hpp"
#include "patch.hpp"
#include "intrinsic_geometry.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
BeamTimoshenko1p<T>::BeamTimoshenko1p(Ptr<UniaxialStress1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamTimoshenko1p: material is null.");
    }
}

// === Matrix Operators =============================================================

template <std::floating_point T> Matrix<T>
BeamTimoshenko1p<T>::strain_matrix(const Patch<T, 1>& /*patch*/,
                                   const BasisDerivs<T, 1>& basis,
                                   const IntrinsicGeometry<T, 1>& ig) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    const Index Q = basis.N_d1(0).rows();
    const Index n = basis.N_d1(0).cols();
    Matrix<T> B(2 * Q, n);


    for (Index q = 0; q < Q; ++q)
    {
        const T gi          = ig.g_inv(0, 0)(q);
        const T G           = ig.chr.Gamma(0, 0, 0)(q);
        const T G2          = G * G;
        const T a11_dot_a11 = ig.a_d1(0, 0).row(q).squaredNorm();
        const T a1_dot_a111 = ig.a(0).row(q).dot(ig.a_d2(0, 0, 0).row(q));
        const T coeff_Nu    = T(4) * G2 - gi * (a11_dot_a11 + a1_dot_a111);
        const T coeff_Nuu   = -T(3) * G;

        // Bending
        // B_b = [ -N_{i|11} ]
        B.row(2 * q    ) = -(basis.N_d2(0, 0).row(q) - G * basis.N_d1(0).row(q));

        // Transverse Shear
        // B_s = [ -(K_b/K_s) g^{11} N_{i|111} ]
        B.row(2 * q + 1) = -ratio * gi
                         * (basis.N_d3(0, 0, 0).row(q)
                            + coeff_Nuu * basis.N_d2(0, 0).row(q)
                            + coeff_Nu  * basis.N_d1(0).row(q));
    }
    return B;
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko1p<T>::constitutive_matrix(const IntrinsicGeometry<T, 1>& ig,
                                                   Index q) const
{
    const T gi = ig.g_inv(0, 0)(q);
    Matrix<T> D = Matrix<T>::Zero(2, 2);

    // D = [ D_b   0   ]
    //     [ 0     D_s ]
    D(0, 0) = material_->bending_stiffness() * gi * gi;
    D(1, 1) = material_->shear_stiffness()   * gi;
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
Matrix<T>
BeamTimoshenko1p<T>::displacement_shape_matrix(const Patch<T, 1>& patch,
                                               const BasisDerivs<T, 1>& basis,
                                               const IntrinsicGeometry<T, 1>& ig) const
{
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    const Index Q = basis.N().rows();
    const Index n = basis.N().cols();
    Matrix<T> Nw(Q, n);


    for (Index q = 0; q < Q; ++q)
    {
        // N_w = [ N_i - (K_b/K_s) g^{11} N_{i|11} ]
        Nw.row(q) = basis.N().row(q) - ratio * ig.g_inv(0, 0)(q) *
                    (basis.N_d2(0, 0).row(q) - ig.chr.Gamma(0, 0, 0)(q) * basis.N_d1(0).row(q));
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T>
BeamTimoshenko1p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                           const BasisDerivs<T, 1>& basis,
                                           const IntrinsicGeometry<T, 1>& /*ig*/) const
{
    // N_rot = [ -N_{i|1} ]
    return -basis.N_d1(0);
}

// === Template Instantiations ========================================================

template class BeamTimoshenko1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko1p<float>;
#endif

} // namespace pyck
