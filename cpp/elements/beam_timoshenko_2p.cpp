#include "beam_timoshenko_2p.hpp"

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

template <std::floating_point T>
Matrix<T> BeamTimoshenko2p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Ni_w = [ Ni 0 ]    (slot 0 = w, slot 1 = θ)
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 2 * n);
    for (Index i = 0; i < n; ++i) {
        Nw.col(2*i) = N[idx::val].col(i);
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko2p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Ni_theta = [ 0 Ni ]
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    Matrix<T> Nth = Matrix<T>::Zero(Q, 2 * n);
    for (Index i = 0; i < n; ++i) {
        Nth.col(2*i + 1) = N[idx::val].col(i);
    }
    return Nth;
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko2p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bb_i = [ 0   Ni,x ]   (kappa = theta,x with theta on slot 1)
    Matrix<T> Bb = Matrix<T>::Zero(Q, 2 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bb(q, 2*i + 1) = N[idx::d1](q, i);
        }
    }
    return Bb;
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko2p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bs_i = [ Ni,x  Ni ]   (gamma = w,x + theta)
    Matrix<T> Bs = Matrix<T>::Zero(Q, 2 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bs(q, 2*i    ) = N[idx::d1](q, i);
            Bs(q, 2*i + 1) = N[idx::val](q, i);
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
