#include "timoshenko_beam_2p.hpp"

namespace pyck
{

template <std::floating_point T>
TimoshenkoBeam2P<T>::TimoshenkoBeam2P(T youngs_modulus,
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
        throw std::invalid_argument("TimoshenkoBeam2P: "
                                    "Young's modulus must be positive.");
    }
    if (A_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam2P: "
                                    "cross-section area must be positive.");
    }
    if (I_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam2P: "
                                    "moment of inertia must be positive.");
    }
    if (G_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam2P: "
                                    "shear modulus must be positive.");
    }
    if (k_ <= 0) {
        throw std::invalid_argument("TimoshenkoBeam2P: "
                                    "shear coefficient must be positive.");
    }
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam2P<T>::shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs[0];
    std::size_t Q = N.rows();
    std::size_t n = N.cols();
    
    Matrix<T> N_mat = Matrix<T>::Zero(2 * Q, 2 * n);
    for (std::size_t q = 0; q < Q; ++q) {
        for (std::size_t i = 0; i < n; ++i) {
            N_mat(2 * q, 2 * i)         = N(q, i); // w
            N_mat(2 * q + 1, 2 * i + 1) = N(q, i); // theta
        }
    }
    return N_mat;
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam2P<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    // Return B for the first quadrature point (unused in assembly, but kept for interface consistency)
    const auto& N = shape_derivs[0];
    const auto& dN_dx = shape_derivs[1];
    std::size_t n = N.cols();
    
    Matrix<T> B = Matrix<T>::Zero(2, 2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        B(0, 2 * i + 1) = dN_dx(0, i);
        B(1, 2 * i)     = dN_dx(0, i);
        B(1, 2 * i + 1) = -N(0, i);
    }
    return B;
}

template <std::floating_point T>
void TimoshenkoBeam2P<T>::compute_local_stiffness(const Patch<T, 1>& patch,
                                                  const ColMatrix<T, 1>& q_points,
                                                  const Vector<T>& q_weights,
                                                  Index span,
                                                  Matrix<T>& stiffness) const
{
    // Need jacobian and derivatives up to 1st order for integration
    auto [shape_fns, jac] = patch.eval_shape_functions(q_points, span, 1);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    std::size_t Q = q_points.rows();
    std::size_t n = shape_fns[0].cols();
    stiffness.setZero(2 * n, 2 * n);

    Matrix<T> D = Matrix<T>::Zero(2, 2);
    D(0, 0) = Kb_;
    D(1, 1) = kGA_;

    for (std::size_t q = 0; q < Q; ++q) {
        Matrix<T> B_q = Matrix<T>::Zero(2, 2 * n);
        for (std::size_t i = 0; i < n; ++i) {
            // Bend (kappa = d(theta)/dx)
            B_q(0, 2 * i + 1) = shape_fns[1](q, i);
            // Shear (gamma = d(w)/dx - theta)
            B_q(1, 2 * i)     = shape_fns[1](q, i);
            B_q(1, 2 * i + 1) = -shape_fns[0](q, i);
        }
        stiffness.noalias() += B_q.transpose() * (D * dV(q)) * B_q;
    }
}

// === Template Instantiations ========================================================

template class TimoshenkoBeam2P<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class TimoshenkoBeam2P<float>;
#endif

} // namespace pyck
