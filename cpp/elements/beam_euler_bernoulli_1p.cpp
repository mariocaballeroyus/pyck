#include "beam_euler_bernoulli_1p.hpp"

namespace pyck
{

template <std::floating_point T>
BeamEulerBernoulli1p<T>::BeamEulerBernoulli1p(Ptr<SlenderBeam1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamEulerBernoulli1p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> BeamEulerBernoulli1p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    return shape_derivs[idx::val];
}

template <std::floating_point T>
Matrix<T> BeamEulerBernoulli1p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // EB kinematics: θ = -dw/dx.
    return -shape_derivs[idx::d1];
}

template <std::floating_point T>
Matrix<T> BeamEulerBernoulli1p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Bb_i = -Ni,xx (kappa = -w,xx)
    return -shape_derivs[idx::d11];
}

template <std::floating_point T>
Matrix<T> BeamEulerBernoulli1p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Euler-Bernoulli has no transverse shear strain.
    const Index Q = shape_derivs[idx::val].rows();
    const Index n = shape_derivs[idx::val].cols();
    return Matrix<T>::Zero(Q, n);
}

// === Template Instantiations ========================================================

template class BeamEulerBernoulli1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamEulerBernoulli1p<float>;
#endif

} // namespace pyck
