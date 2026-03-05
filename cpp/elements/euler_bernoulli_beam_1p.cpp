#include "euler_bernoulli_beam_1p.hpp"

namespace pyck
{

template <std::floating_point T>
EulerBernoulliBeam1P<T>::EulerBernoulliBeam1P(T youngs_modulus,
                                              T section_area,   
                                              T moment_inertia)
    : E_(youngs_modulus), A_(section_area), I_(moment_inertia),
      Kb_(youngs_modulus * moment_inertia)
{
    if (E_ <= 0) {
        throw std::invalid_argument("EulerBernoulliBeam1P: "
                                    "Young's modulus must be positive.");
    }
    if (A_ <= 0) {
        throw std::invalid_argument("EulerBernoulliBeam1P: "
                                    "cross-section area must be positive.");
    }
    if (I_ <= 0) {
        throw std::invalid_argument("EulerBernoulliBeam1P: "
                                    "moment of inertia must be positive.");
    }
}

template <std::floating_point T>
void EulerBernoulliBeam1P<T>::compute_local_stiffness(const Patch<T, 1>& patch,
                                                      const ColMatrix<T, 1>& q_points,
                                                      const Vector<T>& q_weights,
                                                      Index span,
                                                      Matrix<T>& stiffness) const
{
    // Extract shape function values + derivatives (up to 2nd order) and Jacobian
    auto [shape_fns, jac] = patch.eval_shape_functions(q_points, span, 2);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    // Trigger SIMD vectorization by scaling B before the matrix multiplication
    // ( B^T EI B ) dV  --> (B * sqrt(EI * dV))^T * (B * sqrt(EI * dV)) = Beq^T Beq
    Matrix<T> Beq = shape_fns[2];
    Beq.array().colwise() *= (Kb_ * dV).cwiseSqrt().array();
    stiffness.noalias() = Beq.transpose() * Beq;
}

// === Template Instantiations ========================================================

template class EulerBernoulliBeam1P<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class EulerBernoulliBeam1P<float>;
#endif

} // namespace pyck
