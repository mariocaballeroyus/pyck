#include "plate_kirchhoff_love_1p.hpp"

namespace pyck
{

template <std::floating_point T>
PlateKirchhoffLove1p<T>::PlateKirchhoffLove1p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateKirchhoffLove1p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> PlateKirchhoffLove1p<T>::shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Ni = [ Ni ]
    return shape_derivs[idx::val];
}

template <std::floating_point T>
Matrix<T> PlateKirchhoffLove1p<T>::strain_displacement_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N  = shape_derivs;
    const Index Q = N[0].rows(); // number of quadrature points
    const Index n = N[0].cols(); // number of nodes per element

    // Bi = [ -Ni,xx
    //        -Ni,yy
    //        -2*Ni,xy ]
    Matrix<T> B(3 * Q, n);
    for (Index q = 0; q < Q; ++q) {
        // Bending part (kappa_x, kappa_y, kappa_xy)
        B.row(3*q    ) = -N[idx::d11].row(q);
        B.row(3*q + 1) = -N[idx::d22].row(q);
        B.row(3*q + 2) = -T(2) * N[idx::d12].row(q);
    }
    return B;
}

template <std::floating_point T>
void PlateKirchhoffLove1p<T>::compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                                      const Vector<T>& jacobian,
                                                      const Vector<T>& q_weights,
                                                      Matrix<T>& stiffness) const
{
    Matrix<T> B    = this->strain_displacement_matrix(shape_fns); // (3Q, K)
    auto Db = material_->bending_matrix();
    
    const Index Q = q_weights.size(); // number of quadrature points
    const Index K = B.cols();         // number of dofs per element

    // dV = wq * Jq
    Vector<T> dV  = q_weights.cwiseProduct(jacobian); // (Q,)

    // Ke = sum_q{ Bb^T * Db * Bb * dV }
    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        auto Bb = B.middleRows(3 * q, 3);
        stiffness.noalias() += dV[q] * (Bb.transpose() * Db * Bb);
    }
}

// === Template Instantiations ========================================================

template class PlateKirchhoffLove1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateKirchhoffLove1p<float>;
#endif

} // namespace pyck
