#include "beam_timoshenko_2p.hpp"
#include "patch.hpp"
#include "christoffels.hpp"

namespace pyck
{

template <std::floating_point T>
BeamTimoshenko2p<T>::BeamTimoshenko2p(Ptr<SlenderBeam1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamTimoshenko2p: material is null.");
    }
}

template <std::floating_point T> Matrix<T> 
BeamTimoshenko2p<T>::displacement_shape_matrix(const Patch<T, 1>& /*patch*/,
                                               const BasisDerivs<T, 1>& basis,
                                               const LocalFrame<T, 1>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 2 * n);

    // N_w = [ N_i  0 ]
    for (Index i = 0; i < n; ++i)
    {
        Nw.col(2*i) = basis.N.col(i);
    }
    return Nw;
}

template <std::floating_point T> Matrix<T> 
BeamTimoshenko2p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                           const BasisDerivs<T, 1>& basis,
                                           const LocalFrame<T, 1>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Nth = Matrix<T>::Zero(Q, 2 * n);

    // N_rot = [ 0  N_i ]
    for (Index i = 0; i < n; ++i)
    {
        Nth.col(2*i + 1) = basis.N.col(i);
    }
    return Nth;
}

template <std::floating_point T> T 
BeamTimoshenko2p<T>::bending_constitutive(const LocalFrame<T, 1>& local, 
                                          Index q) const
{
    // D_b = EI g^{11}
    return material_->bending_stiffness() * local.g_inv_11(q);
}

template <std::floating_point T> T 
BeamTimoshenko2p<T>::shear_constitutive(const LocalFrame<T, 1>& local,
                                        Index q) const
{
    // D_s = kGA g^{11}
    return material_->shear_stiffness() * local.g_inv_11(q);
}

template <std::floating_point T> Matrix<T> 
BeamTimoshenko2p<T>::bending_strain_matrix(const Patch<T, 1>& patch,
                                           const BasisDerivs<T, 1>& basis,
                                           const LocalFrame<T, 1>& local) const
{
    auto chr = eval_christoffel(local);
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Bb = Matrix<T>::Zero(Q, 2 * n);

    // B_b = [ 0  N_{i|1} ]
    for (Index q = 0; q < Q; ++q)
    {
        for (Index i = 0; i < n; ++i) 
        {
            Bb(q, 2*i + 1) = basis.N_u(q, i) - chr.G1_11(q) * basis.N(q, i);
        }
    }
    return Bb;
}

template <std::floating_point T> Matrix<T> 
BeamTimoshenko2p<T>::shear_strain_matrix(const Patch<T, 1>& /*patch*/,
                                         const BasisDerivs<T, 1>& basis,
                                         const LocalFrame<T, 1>& /*local*/) const
{
    const Index Q = basis.N.rows();
    const Index n = basis.N.cols();
    Matrix<T> Bs = Matrix<T>::Zero(Q, 2 * n);

    // B_s = [ N_{i|1}  N_i ]
    for (Index q = 0; q < Q; ++q)
    {
        for (Index i = 0; i < n; ++i) 
        {
            Bs(q, 2*i    ) = basis.N_u(q, i);
            Bs(q, 2*i + 1) = basis.N(q, i);
        }
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class BeamTimoshenko2p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko2p<float>;
#endif

} // namespace pyck
