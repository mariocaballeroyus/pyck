#include "plate_reissner_mindlin_1p.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlin1p<T>::PlateReissnerMindlin1p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlin1p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin1p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    // Ni_w = [ Ni - (Kb/Ks) * (Ni,xx + Ni,yy) ]
    return N[idx::val] - ratio * (N[idx::d11] + N[idx::d22]);
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin1p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::d1].rows();
    const Index n = N[idx::d1].cols();
    Matrix<T> Nphi(2 * Q, n);

    // Ni_phi = [ -Ni,x 
    //            -Ni,y ]
    for (Index q = 0; q < Q; ++q) {
        Nphi.row(2*q    ) = -N[idx::d1].row(q);
        Nphi.row(2*q + 1) = -N[idx::d2].row(q);
    }
    return Nphi;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin1p<T>::strain_displacement_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[0].rows();
    const Index n = N[0].cols();
    T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    // Bi = [ Ni,xx                          ]  (kappa_x)
    //      [ Ni,yy                          ]  (kappa_y)
    //      [ 2*Ni,xy                        ]  (kappa_xy)
    //      [ -Kb/Ks * (Ni,xxx + Ni,xyy)     ]  (gamma_xz correction)
    //      [ -Kb/Ks * (Ni,xxy + Ni,yyy)     ]  (gamma_yz correction)
    Matrix<T> B(5 * Q, n);
    for (Index q = 0; q < Q; ++q) {
        // Bending part (curvatures)
        B.row(5*q    ) = -N[idx::d11].row(q);
        B.row(5*q + 1) = -N[idx::d22].row(q);
        B.row(5*q + 2) = -T(2) * N[idx::d12].row(q);
        // Shear part: gamma = -(Kb/Ks) * grad(laplacian(w_b))
        B.row(5*q + 3) = -ratio * (N[idx::d111].row(q) + N[idx::d122].row(q));
        B.row(5*q + 4) = -ratio * (N[idx::d112].row(q) + N[idx::d222].row(q));
    }
    return B;
}

template <std::floating_point T>
void PlateReissnerMindlin1p<T>::compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                                        const Vector<T>& jacobian,
                                                        const Vector<T>& q_weights,
                                                        Matrix<T>& stiffness) const
{
    Matrix<T> B   = this->strain_displacement_matrix(shape_fns); // (5Q, K)
    auto Db       = material_->bending_matrix();                 // (3, 3)
    auto Ds       = material_->shear_matrix();                   // (2, 2)

    const Index Q = q_weights.size(); // number of quadrature points
    const Index K = B.cols();         // number of dofs per element

    // dV = wq * Jq
    Vector<T> dV  = q_weights.cwiseProduct(jacobian);

    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        auto Bb = B.middleRows(5 * q,     3);
        auto Bs = B.middleRows(5 * q + 3, 2);
        stiffness.noalias() += dV[q] * (Bb.transpose() * Db * Bb
                                      + Bs.transpose() * Ds * Bs);
    }
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin1p<float>;
#endif

} // namespace pyck
