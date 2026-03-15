#include "timoshenko_beam_2p.hpp"

namespace pyck
{

template <std::floating_point T>
TimoshenkoBeam2p<T>::TimoshenkoBeam2p(Ptr<SlenderBeam1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("TimoshenkoBeam2p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam2p<T>::shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    std::size_t Q = N[idx::fn].rows();
    std::size_t n = N[idx::fn].cols();
    
    Matrix<T> N_mat = Matrix<T>::Zero(2 * Q, 2 * n);
    for (std::size_t q = 0; q < Q; ++q) {
        for (std::size_t i = 0; i < n; ++i) {
            N_mat(2 * q, 2 * i)         = N[idx::fn](q, i); // w
            N_mat(2 * q + 1, 2 * i + 1) = N[idx::fn](q, i); // theta
        }
    }
    return N_mat;
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam2p<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    // Return B for the first quadrature point (unused in assembly, but kept for interface consistency)
    const auto& N = shape_derivs;
    std::size_t n = N[idx::fn].cols();
    
    Matrix<T> B = Matrix<T>::Zero(2, 2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        B(0, 2 * i + 1) = N[idx::u](0, i);
        B(1, 2 * i)     = N[idx::u](0, i);
        B(1, 2 * i + 1) = -N[idx::fn](0, i);
    }
    return B;
}

template <std::floating_point T>
void TimoshenkoBeam2p<T>::compute_local_stiffness(const Patch<T, 1>& patch,
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
    D(0, 0) = material_->bending_stiffness();
    D(1, 1) = material_->shear_stiffness();

    for (std::size_t q = 0; q < Q; ++q) {
        const auto& N = shape_fns;
        Matrix<T> B_q = Matrix<T>::Zero(2, 2 * n);

        for (std::size_t i = 0; i < n; ++i) {
            // Bend (kappa = d(theta)/dx)
            B_q(0, 2 * i + 1) = N[idx::u](q, i);
            // Shear (gamma = d(w)/dx - theta)
            B_q(1, 2 * i)     = N[idx::u](q, i);
            B_q(1, 2 * i + 1) = -N[idx::fn](q, i);
        }
        stiffness.noalias() += B_q.transpose() * (D * dV(q)) * B_q;
    }
}

// === Template Instantiations ========================================================

template class TimoshenkoBeam2p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class TimoshenkoBeam2p<float>;
#endif

} // namespace pyck
