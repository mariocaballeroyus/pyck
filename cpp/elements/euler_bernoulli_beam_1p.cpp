#include "euler_bernoulli_beam_1p.hpp"

namespace pyck
{

template <std::floating_point T>
EulerBernoulliBeam1p<T>::EulerBernoulliBeam1p(Ptr<SlenderBeam1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("EulerBernoulliBeam1p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> EulerBernoulliBeam1p<T>::shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    return shape_derivs[0];
}

template <std::floating_point T>
Matrix<T> EulerBernoulliBeam1p<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    return shape_derivs[idx::uu];
}

template <std::floating_point T>
void EulerBernoulliBeam1p<T>::compute_local_stiffness(const Patch<T, 1>& patch,
                                                      const ColMatrix<T, 1>& q_points,
                                                      const Vector<T>& q_weights,
                                                      Index span,
                                                      Matrix<T>& stiffness) const
{
    // Need jacobian and derivatives up to 2nd order for integration
    auto [shape_fns, jac] = patch.eval_shape_functions(q_points, span, 2);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    // Trigger SIMD vectorization by scaling B before the matrix multiplication
    // ( B^T EI B ) dV  --> (B * sqrt(EI * dV))^T * (B * sqrt(EI * dV)) = Beq^T Beq
    Matrix<T> B = this->strain_displacement_matrix(shape_fns);
    Matrix<T> Beq = B;
    Beq.array().colwise() *= (material_->bending_stiffness() * dV).cwiseSqrt().array();
    stiffness.noalias() = Beq.transpose() * Beq;
}

// === Template Instantiations ========================================================

template class EulerBernoulliBeam1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class EulerBernoulliBeam1p<float>;
#endif

} // namespace pyck
