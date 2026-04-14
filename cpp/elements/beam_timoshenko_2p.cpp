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
Matrix<T> BeamTimoshenko2p<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    const Index Q = shape_derivs[idx::val].rows(); // number of quadrature points
    const Index n = shape_derivs[idx::val].cols(); // number of dofs per element

    // Bi = [ 0      N1,x
    //        N1,x   N1 ]
    Matrix<T> B = Matrix<T>::Zero(2 * Q, 2 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            // Bending part (kappa_x = theta,x)
            B(2*q,     2*i + 1) =  shape_derivs[idx::d1](q, i);
            // Shear part (gamma_xz = w,x + theta)
            B(2*q + 1, 2*i    ) =  shape_derivs[idx::d1](q, i);
            B(2*q + 1, 2*i + 1) =  shape_derivs[idx::val](q, i);
        }
    }
    return B;
}

template <std::floating_point T>
void BeamTimoshenko2p<T>::compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                                  const Vector<T>& jacobian,
                                                  const Vector<T>& q_weights,
                                                  Matrix<T>& stiffness) const
{
    Matrix<T> B   = this->strain_displacement_matrix(shape_fns); // (2Q, 2n)
    T Kb          = material_->bending_stiffness();              // Kb = EI
    T Ks          = material_->shear_stiffness();                // Ks = kGA

    const Index Q = q_weights.size(); // number of quadrature points
    const Index K = B.cols();         // number of dofs per element

    // dV = wq * Jq
    Vector<T> dV = q_weights.cwiseProduct(jacobian); // (Q,)

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

template class BeamTimoshenko2p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamTimoshenko2p<float>;
#endif

} // namespace pyck
