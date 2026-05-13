#include "beam_euler_bernoulli_1p.hpp"
#include "patch.hpp"
#include "christoffels.hpp"

namespace pyck
{

template <std::floating_point T>
BeamEulerBernoulli1p<T>::BeamEulerBernoulli1p(Ptr<SlenderBeam1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamEulerBernoulli1p: "
                                    "material is null.");
    }
}

template <std::floating_point T> Matrix<T>
BeamEulerBernoulli1p<T>::displacement_shape_matrix(const Patch<T, 1>& /*patch*/,
                                                   const BasisDerivs<T, 1>& basis,
                                                   const LocalFrame<T, 1>& /*local*/) const
{
    // N_w = [ N_i ]
    return basis.N;
}

template <std::floating_point T> Matrix<T>
BeamEulerBernoulli1p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                               const BasisDerivs<T, 1>& basis,
                                               const LocalFrame<T, 1>& /*local*/) const
{
    // N_rot = [ -N_{i|1} ]
    return -basis.N_u;
}

template <std::floating_point T>
T BeamEulerBernoulli1p<T>::bending_constitutive(const LocalFrame<T, 1>& local,
                                                Index q) const
{
    const T gi = local.g_inv_11(q);

    // D_b = EI (g^{11})^2
    return material_->bending_stiffness() * gi * gi;
}

template <std::floating_point T> Matrix<T>
BeamEulerBernoulli1p<T>::bending_strain_matrix(const Patch<T, 1>& patch,
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
BeamEulerBernoulli1p<T>::shear_strain_matrix(const Patch<T, 1>& /*patch*/,
                                             const BasisDerivs<T, 1>& basis,
                                             const LocalFrame<T, 1>& /*local*/) const
{
    // Normality assumption (zero transverse shear strain)
    return Matrix<T>::Zero(0, basis.N.cols());
}

// === Template Instantiations ========================================================

template class BeamEulerBernoulli1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamEulerBernoulli1p<float>;
#endif

} // namespace pyck
