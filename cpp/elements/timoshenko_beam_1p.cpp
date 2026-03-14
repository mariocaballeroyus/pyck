#include "timoshenko_beam_1p.hpp"

namespace pyck
{

template <std::floating_point T>
TimoshenkoBeam1P<T>::TimoshenkoBeam1P(T youngs_modulus,
                                      T section_area,   
                                      T moment_inertia,
                                      T shear_modulus,
                                      T shear_coefficient)
    : E_(youngs_modulus), A_(section_area), I_(moment_inertia),
      G_(shear_modulus), k_(shear_coefficient),
      kGA_(shear_coefficient * shear_modulus * section_area), 
      Kb_(youngs_modulus * moment_inertia)
{
    if (E_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam1P: "
                                    "Young's modulus must be positive.");
    }
    if (A_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam1P: "
                                    "cross-section area must be positive.");
    }
    if (I_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam1P: "
                                    "moment of inertia must be positive.");
    }
    if (G_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam1P: "
                                    "shear modulus must be positive.");
    }
    if (k_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam1P: "
                                    "shear coefficient must be positive.");
    }
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam1P<T>::shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    // Evaluates the generalized shape matrix N_tilde used for load vectors.
    // N_tilde = [N - (Kb/Ks)*N'']
    const auto& N = shape_derivs[0];
    const auto& d2N_dx2 = shape_derivs[2]; // Need 2nd derivatives!
    
    std::size_t Q = N.rows();
    std::size_t n = N.cols();
    
    Matrix<T> N_mat(Q, n);
    T ratio = Kb_ / kGA_;
    for (std::size_t q = 0; q < Q; ++q) {
        for (std::size_t i = 0; i < n; ++i) {
            N_mat(q, i) = N(q, i) - ratio * d2N_dx2(q, i);
        }
    }
    return N_mat;
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam1P<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& d2N_dx2 = shape_derivs[2]; // B_bend
    const auto& d3N_dx3 = shape_derivs[3]; // B_shear
    std::size_t n = d2N_dx2.cols();
    
    Matrix<T> B(2, n);
    T ratio = Kb_ / kGA_;
    for (std::size_t i = 0; i < n; ++i) {
        B(0, i) = d2N_dx2(0, i);
        B(1, i) = -ratio * d3N_dx3(0, i);
    }
    return B;
}

template <std::floating_point T>
void TimoshenkoBeam1P<T>::compute_local_stiffness(const Patch<T, 1>& patch,
                                                  const ColMatrix<T, 1>& q_points,
                                                  const Vector<T>& q_weights,
                                                  Index span,
                                                  Matrix<T>& stiffness) const
{
    // Need jacobian and derivatives up to 3rd order for integration!
    auto [shape_fns, jac] = patch.eval_shape_functions(q_points, span, 3);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    std::size_t Q = q_points.rows();
    std::size_t n = shape_fns[0].cols();
    stiffness.setZero(n, n);

    Matrix<T> D = Matrix<T>::Zero(2, 2);
    D(0, 0) = Kb_;
    D(1, 1) = kGA_;
    
    T ratio = Kb_ / kGA_;

    for (std::size_t q = 0; q < Q; ++q) {
        Matrix<T> B_q = Matrix<T>::Zero(2, n);
        for (std::size_t i = 0; i < n; ++i) {
            // Bend (N'')
            B_q(0, i) = shape_fns[2](q, i);
            // Shear ( -(Kb/Ks) * N''' )
            B_q(1, i) = -ratio * shape_fns[3](q, i);
        }
        stiffness.noalias() += B_q.transpose() * (D * dV(q)) * B_q;
    }
}

// === Template Instantiations ========================================================

template class TimoshenkoBeam1P<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class TimoshenkoBeam1P<float>;
#endif

} // namespace pyck
