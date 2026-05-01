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
Matrix<T> PlateReissnerMindlin3p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * n);

    // Ni_w = [ Ni 0 0 ]
    for (Index i = 0; i < n; ++i) {
        Nw.col(3*i) = N[idx::val].col(i);
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin3p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * n);

    // Ni_phi = [ 0 Ni 0
    //            0 0  Ni ]
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Nphi(2*q,     3*i + 1) = N[idx::val](q, i);
            Nphi(2*q + 1, 3*i + 2) = N[idx::val](q, i);
        }
    }
    return Nphi;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin3p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bb_i (3 x 3): kappa = grad(theta) with theta on slots 1, 2
    //   row 0 (kappa_x):    [ 0   Ni,x  0    ]
    //   row 1 (kappa_y):    [ 0   0     Ni,y ]
    //   row 2 (2 kappa_xy): [ 0   Ni,y  Ni,x ]
    Matrix<T> Bb = Matrix<T>::Zero(3 * Q, 3 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bb(3*q,     3*i + 1) = N[idx::d1](q, i);
            Bb(3*q + 1, 3*i + 2) = N[idx::d2](q, i);
            Bb(3*q + 2, 3*i + 1) = N[idx::d2](q, i);
            Bb(3*q + 2, 3*i + 2) = N[idx::d1](q, i);
        }
    }
    return Bb;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlin3p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bs_i (2 x 3): gamma_xz = w,x + theta_x, gamma_yz = w,y + theta_y
    //   row 0 (gamma_x): [ Ni,x   Ni    0   ]
    //   row 1 (gamma_y): [ Ni,y   0     Ni  ]
    Matrix<T> Bs = Matrix<T>::Zero(2 * Q, 3 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bs(2*q,     3*i    ) = N[idx::d1](q, i);
            Bs(2*q,     3*i + 1) = N[idx::val](q, i);
            Bs(2*q + 1, 3*i    ) = N[idx::d2](q, i);
            Bs(2*q + 1, 3*i + 2) = N[idx::val](q, i);
        }
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlin3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlin3p<float>;
#endif

} // namespace pyck
