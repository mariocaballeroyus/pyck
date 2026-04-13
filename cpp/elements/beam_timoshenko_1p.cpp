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
Matrix<T> BeamTimoshenko1p<T>::transverse_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    // Effective shape function for total deflection: Ñ_i = N_i - (Kb/Ks) N_i,xx
    // This accounts for w = w_b - (Kb/Ks) w_b,xx in the load-vector integral.
    return N[idx::val] - ratio * N[idx::d11];
}

template <std::floating_point T>
Matrix<T> BeamTimoshenko1p<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    T ratio       = material_->bending_stiffness() / material_->shear_stiffness();
    const Index Q = N[idx::val].rows(); // number of quadrature points
    const Index n = N[idx::val].cols(); // number of dofs per element

    // Bi = [ -Ni,xx 
    //        -ratio * Ni,xxx ]
    Matrix<T> B(2 * Q, n);
    for (Index q = 0; q < Q; ++q) {
        // Bending part (kappa_x = -w,xx)
        B.row(2*q    ) = -N[idx::d11].row(q);
        // Shear part (gamma_xz = -ratio * w,xxx)
        B.row(2*q + 1) = -ratio * N[idx::d111].row(q);
    }
    return B;
}

template <std::floating_point T>
void BeamTimoshenko1p<T>::compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                                  const Vector<T>& jacobian,
                                                  const Vector<T>& q_weights,
                                                  Matrix<T>& stiffness) const
{
    Matrix<T> B   = this->strain_displacement_matrix(shape_fns); // (2Q, n)
    T Kb          = material_->bending_stiffness();              // Kb = EI
    T Ks          = material_->shear_stiffness();                // Ks = kGA
    
    const Index Q = q_weights.size(); // number of quadrature points
    const Index K = B.cols();         // number of dofs per element

    // dV = wq * Jq
    Vector<T> dV  = q_weights.cwiseProduct(jacobian); // (Q,)

    // Ke = sum_q{ Bb^T * Kb * Bb * dV  +  Bs^T * Ks * Bs * dV }
    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        auto Bb = B.row(2 * q);
        auto Bs = B.row(2 * q + 1);
        stiffness.noalias() += dV[q] * (Kb * Bb.transpose() * Bb
                                      + Ks * Bs.transpose() * Bs);
    }
}

// === Template Instantiations ========================================================

template class BeamTimoshenko1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko1p<float>;
#endif

} // namespace pyck
