#include "plate_reissner_mindlin_3p.hpp"
#include "patch.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlin3p<T>::PlateReissnerMindlin3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlin3p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin3p<T>::shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const std::size_t Q = shape_derivs[idx::val].rows(); // number of quadrature points
    const std::size_t n = shape_derivs[idx::val].cols(); // number of nodes per element
    
    // Ni = [ Ni 0  0
    //        0  Ni 0
    //        0  0  Ni ]
    Matrix<T> N = Matrix<T>::Zero(3 * Q, 3 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            N(3*q,     3*i    ) = shape_derivs[idx::val](q, i);
            N(3*q + 1, 3*i + 1) = shape_derivs[idx::val](q, i);
            N(3*q + 2, 3*i + 2) = shape_derivs[idx::val](q, i);
        }
    }
    return N;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin3p<T>::strain_displacement_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N  = shape_derivs;
    const std::size_t Q = N[idx::val].rows(); // number of quadrature points
    const std::size_t n = N[idx::val].cols(); // number of nodes per element
    
    // Bi = [ 0    -Ni,x  0
    //        0     0    -Ni,y
    //        0    -Ni,y -Ni,x
    //       Ni,x   Ni    0
    //       Ni,y   0     Ni   ]
    Matrix<T> B = Matrix<T>::Zero(5 * Q, 3 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            // Bending part (kappa_x, kappa_y, kappa_xy)
            B(5*q,     3*i + 1) =  N[idx::d1](q, i);
            B(5*q + 1, 3*i + 2) =  N[idx::d2](q, i);
            B(5*q + 2, 3*i + 1) =  N[idx::d2](q, i);
            B(5*q + 2, 3*i + 2) =  N[idx::d1](q, i);
            // Shear part (gamma_xz = w,x + theta_x, gamma_yz = w,y + theta_y)
            B(5*q + 3, 3*i    ) =  N[idx::d1](q, i);
            B(5*q + 3, 3*i + 1) =  N[idx::val](q, i);
            B(5*q + 4, 3*i    ) =  N[idx::d2](q, i);
            B(5*q + 4, 3*i + 2) =  N[idx::val](q, i);
        }
    }
    return B;
}

template <std::floating_point T>
void PlateReissnerMindlin3p<T>::compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                                       const Vector<T>& jacobian,
                                                       const Vector<T>& q_weights,
                                                       Matrix<T>& stiffness) const
{
    Matrix<T> B   = this->strain_displacement_matrix(shape_fns); // (5Q, 3K)
    auto Db       = material_->bending_matrix();
    auto Ds       = material_->shear_matrix();

    const Index Q = q_weights.size(); // number of quadrature points
    const Index K = B.cols();         // number of dofs per element

    // dV = wq * Jq
    Vector<T> dV  = q_weights.cwiseProduct(jacobian); // (Q,)

    // Ke = sum_q{ Bb^T * Db * Bb * dV  +  Bs^T * Ds * Bs * dV }
    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        auto Bb = B.middleRows(5 * q,     3);
        auto Bs = B.middleRows(5 * q + 3, 2);
        stiffness.noalias() += dV[q] * (Bb.transpose() * Db * Bb
                                      + Bs.transpose() * Ds * Bs);
    }
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin3p<float>;
#endif

} // namespace pyck
