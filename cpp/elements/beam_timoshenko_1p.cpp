#include "beam_timoshenko_1p.hpp"
#include "patch.hpp"
#include "christoffels.hpp"

namespace pyck
{

template <std::floating_point T>
BeamTimoshenko1p<T>::BeamTimoshenko1p(Ptr<SlenderBeam1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamTimoshenko1p: material is null.");
    }
}

template <std::floating_point T> Matrix<T> 
BeamTimoshenko1p<T>::displacement_shape_matrix(const Patch<T, 1>& patch,
                                               const BasisDerivs<T, 1>& basis,
                                               const LocalFrame<T, 1>& local) const
{
    auto chr = eval_christoffel(local);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nw(Q, n);

    // N_w = [ N_i - (K_b/K_s) g^{11} N_{i|11} ]
    for (Index q = 0; q < Q; ++q)
    {
        Nw.row(q) = basis.N.row(q) - ratio * local.g_inv_11(q) *
                    (basis.N_uu.row(q) - chr.G1_11(q) * basis.N_u.row(q));
    }
    return Nw;
}

template <std::floating_point T> Matrix<T> 
BeamTimoshenko1p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                           const BasisDerivs<T, 1>& basis,
                                           const LocalFrame<T, 1>& /*local*/) const
{
    // N_rot = [ -N_{i|1} ]
    return -basis.N_u;
}

template <std::floating_point T>
T BeamTimoshenko1p<T>::bending_constitutive(const LocalFrame<T, 1>& local, Index q) const
{
    // D_b = EI (g^{11})^2
    const T gi = local.g_inv_11(q);
    return material_->bending_stiffness() * gi * gi;
}

template <std::floating_point T>
T BeamTimoshenko1p<T>::shear_constitutive(const LocalFrame<T, 1>& local, Index q) const
{
    // D_s = kGA g^{11}
    return material_->shear_stiffness() * local.g_inv_11(q);
}

template <std::floating_point T> Matrix<T>
BeamTimoshenko1p<T>::bending_strain_matrix(const Patch<T, 1>& patch,
                                           const BasisDerivs<T, 1>& basis,
                                           const LocalFrame<T, 1>& local) const
{
    auto chr = eval_christoffel(local);
    const Index Q = basis.N_u.rows();
    const Index n = basis.N_u.cols();
    Matrix<T> Bb(Q, n);

    // B_b = [ -N_{i|11} ]
    for (Index q = 0; q < Q; ++q)
    {
        Bb.row(q) = -(basis.N_uu.row(q) - chr.G1_11(q) * basis.N_u.row(q));
    }
    return Bb;
}

template <std::floating_point T> Matrix<T> 
BeamTimoshenko1p<T>::shear_strain_matrix(const Patch<T, 1>& patch,
                                         const BasisDerivs<T, 1>& basis,
                                         const LocalFrame<T, 1>& local) const
{
    auto chr = eval_christoffel(local);
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    const Index Q = basis.N_u.rows();
    const Index n = basis.N_u.cols();
    Matrix<T> Bs(Q, n);

    // B_s = [ -(K_b/K_s) g^{11} N_{i|111} ]
    for (Index q = 0; q < Q; ++q)
    {
        const T gi   = local.g_inv_11(q);
        const T G    = chr.G1_11(q);
        const T G2   = G * G;
        const T a11_dot_a11   = local.a11.row(q).squaredNorm();
        const T a1_dot_a111   = local.a1.row(q).dot(local.a111.row(q));
        
        const T coeff_Nu      = T(4) * G2 - gi * (a11_dot_a11 + a1_dot_a111);
        const T coeff_Nuu     = -T(3) * G;

        Bs.row(q) = -ratio * gi
                  * (basis.N_uuu.row(q)
                     + coeff_Nuu * basis.N_uu.row(q)
                     + coeff_Nu  * basis.N_u.row(q));
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class BeamTimoshenko1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko1p<float>;
#endif

} // namespace pyck
