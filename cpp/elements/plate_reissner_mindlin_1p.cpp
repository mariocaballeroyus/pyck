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
Matrix<T> PlateReissnerMindlin1p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bb_i = [ -Ni,xx     ]  (kappa_x)
    //        [ -Ni,yy     ]  (kappa_y)
    //        [ -2 Ni,xy   ]  (2 kappa_xy)
    Matrix<T> Bb(3 * Q, n);
    for (Index q = 0; q < Q; ++q) {
        Bb.row(3*q    ) = -N[idx::d11].row(q);
        Bb.row(3*q + 1) = -N[idx::d22].row(q);
        Bb.row(3*q + 2) = -T(2) * N[idx::d12].row(q);
    }
    return Bb;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin1p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    // Bs_i = [ -(Kb/Ks)(Ni,xxx + Ni,xyy) ]
    //        [ -(Kb/Ks)(Ni,xxy + Ni,yyy) ]
    Matrix<T> Bs(2 * Q, n);
    for (Index q = 0; q < Q; ++q) {
        Bs.row(2*q    ) = -ratio * (N[idx::d111].row(q) + N[idx::d122].row(q));
        Bs.row(2*q + 1) = -ratio * (N[idx::d112].row(q) + N[idx::d222].row(q));
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin1p<float>;
#endif

} // namespace pyck
