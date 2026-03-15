#include "timoshenko_beam_1p.hpp"

namespace pyck
{

template <std::floating_point T>
TimoshenkoBeam1p<T>::TimoshenkoBeam1p(Ptr<SlenderBeam1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("TimoshenkoBeam1p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam1p<T>::shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    // Evaluates the generalized shape matrix N_tilde used for load vectors.
    // N_tilde = [N - (Kb/Ks)*N'']
    const auto& N = shape_derivs;
    
    std::size_t Q = N[idx::fn].rows();
    std::size_t n = N[idx::fn].cols();
    
    Matrix<T> N_mat(Q, n);
    T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    for (std::size_t q = 0; q < Q; ++q) {
        for (std::size_t i = 0; i < n; ++i) {
            N_mat(q, i) = N[idx::fn](q, i) - ratio * N[idx::uu](q, i);
        }
    }
    return N_mat;
}

template <std::floating_point T>
Matrix<T> TimoshenkoBeam1p<T>::strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    std::size_t n = N[idx::uu].cols();
    
    Matrix<T> B(2, n);
    T ratio = material_->bending_stiffness() / material_->shear_stiffness();
    for (std::size_t i = 0; i < n; ++i) {
        B(0, i) = N[idx::uu](0, i);
        B(1, i) = -ratio * N[idx::uuu](0, i);
    }
    return B;
}

template <std::floating_point T>
void TimoshenkoBeam1p<T>::compute_local_stiffness(const Patch<T, 1>& patch,
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
    D(0, 0) = material_->bending_stiffness();
    D(1, 1) = material_->shear_stiffness();
    
    T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    for (std::size_t q = 0; q < Q; ++q) {
        const auto& N = shape_fns;
        Matrix<T> B_q = Matrix<T>::Zero(2, n);

        for (std::size_t i = 0; i < n; ++i) {
            // Bend (N'')
            B_q(0, i) = N[idx::uu](q, i);
            // Shear ( -(Kb/Ks) * N''' )
            B_q(1, i) = -ratio * N[idx::uuu](q, i);
        }
        stiffness.noalias() += B_q.transpose() * (D * dV(q)) * B_q;
    }
}

// === Template Instantiations ========================================================

template class TimoshenkoBeam1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class TimoshenkoBeam1p<float>;
#endif

} // namespace pyck
