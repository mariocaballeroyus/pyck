#include "beam_timoshenko_1p.hpp"

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

template <std::floating_point T>
Matrix<T> BeamTimoshenko1p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    // Effective shape function for total deflection: Ñ_i = N_i - (Kb/Ks) N_i,xx
    // This accounts for w = w_b - (Kb/Ks) w_b,xx in the load-vector integral.
    return N[idx::val] - ratio * N[idx::d11];
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko1p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Same derivation as 2D RM-1p: θ = -dw_b/dx.
    return -shape_derivs[idx::d1];
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko1p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Bb_i = -Ni,xx (kappa = -w,xx)
    return -shape_derivs[idx::d11];
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko1p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // gamma = -(Kb/Ks) w,xxx
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    return -ratio * shape_derivs[idx::d111];
}

// === Template Instantiations ========================================================

template class BeamTimoshenko1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko1p<float>;
#endif

} // namespace pyck
