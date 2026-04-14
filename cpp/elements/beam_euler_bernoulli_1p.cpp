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
Matrix<T> BeamEulerBernoulli1p<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    // Bi = [ -Ni,xx ]
    return -shape_derivs[idx::d11];
}

template <std::floating_point T>
void BeamEulerBernoulli1p<T>::compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                                     const Vector<T>& jacobian,
                                                     const Vector<T>& q_weights,
                                                     Matrix<T>& stiffness) const
{
    Matrix<T> B   = this->strain_displacement_matrix(shape_fns); // (Q, n)
    T Kb          = material_->bending_stiffness();              // Kb = EI
    
    const Index Q = q_weights.size(); // number of quadrature points
    const Index K = B.cols();         // number of dofs per element

    // dV = wq * Jq
    Vector<T> dV  = q_weights.cwiseProduct(jacobian); // (Q,)

    // Ke = sum_q{ Bb^T * Kb * Bb * dV }
    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        auto Bb = B.row(q);
        stiffness.noalias() += dV[q] * (Kb * Bb.transpose() * Bb);
    }
}

// === Template Instantiations ========================================================

template class BeamEulerBernoulli1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamEulerBernoulli1p<float>;
#endif

} // namespace pyck
