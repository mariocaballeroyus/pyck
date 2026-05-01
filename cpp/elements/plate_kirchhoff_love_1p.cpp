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
Matrix<T> PlateKirchhoffLove1p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Ni_w = Ni
    return shape_derivs[idx::val];
}

template <std::floating_point T>
Matrix<T> PlateKirchhoffLove1p<T>::rotation_shape_matrix(
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
Matrix<T> PlateKirchhoffLove1p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bb_i = [ -Ni,xx     ]   kappa_x
    //        [ -Ni,yy     ]   kappa_y
    //        [ -2 Ni,xy   ]   2 kappa_xy
    Matrix<T> Bb(3 * Q, n);
    for (Index q = 0; q < Q; ++q) {
        Bb.row(3*q    ) = -N[idx::d11].row(q);
        Bb.row(3*q + 1) = -N[idx::d22].row(q);
        Bb.row(3*q + 2) = -T(2) * N[idx::d12].row(q);
    }
    return Bb;
}

template <std::floating_point T>
Matrix<T> PlateKirchhoffLove1p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Kirchhoff-Love has no transverse shear strain.
    const Index Q = shape_derivs[idx::val].rows();
    const Index n = shape_derivs[idx::val].cols();
    return Matrix<T>::Zero(2 * Q, n);
}

template <std::floating_point T>
Matrix<T> PlateKirchhoffLove1p<T>::transverse_shear_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // KL has no shear strain; recover q via equilibrium q = -div(m):
    //   q_x = -(m_x,x + m_xy,y) = -D (Ni,xxx + (2-nu) Ni,xyy)
    //   q_y = -(m_xy,x + m_y,y) = -D ((2-nu) Ni,xxy + Ni,yyy)
    const auto& N = shape_derivs;
    const Index Q = N[idx::d1].rows();
    const Index n = N[idx::d1].cols();
    const T D = material_->bending_stiffness();
    const T nu = material_->poisson_ratio();

    Matrix<T> Nq(2 * Q, n);
    for (Index q = 0; q < Q; ++q) {
        Nq.row(2*q    ) = -D * (N[idx::d111].row(q) + (T(2) - nu) * N[idx::d122].row(q));
        Nq.row(2*q + 1) = -D * ((T(2) - nu) * N[idx::d112].row(q) + N[idx::d222].row(q));
    }
    return Nq;
}

// === Template Instantiations ========================================================

template class PlateKirchhoffLove1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateKirchhoffLove1p<float>;
#endif

} // namespace pyck
