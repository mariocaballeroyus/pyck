#include "plate_reissner_mindlin_displ_3p.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlinDispl3p<T>::PlateReissnerMindlinDispl3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlinDispl3p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl3p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * n);

    // Nw_i = [ N_i  N_i  N_i ]   (w = w_b + w_s1 + w_s2)
    for (Index i = 0; i < n; ++i) {
        Nw.col(3 * i    ) = N[idx::val].col(i);
        Nw.col(3 * i + 1) = N[idx::val].col(i);
        Nw.col(3 * i + 2) = N[idx::val].col(i);
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl3p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * n);

    // Nphi_i = [ -N_i,x   0       -N_i,x
    //            -N_i,y  -N_i,y    0     ]
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Nphi(2 * q,     3 * i)     = -N[idx::d1](q, i);
            Nphi(2 * q,     3 * i + 2) = -N[idx::d1](q, i);
            Nphi(2 * q + 1, 3 * i)     = -N[idx::d2](q, i);
            Nphi(2 * q + 1, 3 * i + 1) = -N[idx::d2](q, i);
        }
    }
    return Nphi;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl3p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bb_i (3 x 3): kappa = L phi, phi = -grad(w_b) - (w_s2,x, w_s1,y)
    //   row 0 (kappa_x):    [ -N_i,xx   0          -N_i,xx          ]
    //   row 1 (kappa_y):    [ -N_i,yy  -N_i,yy      0               ]
    //   row 2 (2 kappa_xy): [ -2 N_i,xy -N_i,xy    -N_i,xy           ]
    Matrix<T> Bb = Matrix<T>::Zero(3 * Q, 3 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bb(3 * q,     3 * i)     = -N[idx::d11](q, i);
            Bb(3 * q,     3 * i + 2) = -N[idx::d11](q, i);

            Bb(3 * q + 1, 3 * i)     = -N[idx::d22](q, i);
            Bb(3 * q + 1, 3 * i + 1) = -N[idx::d22](q, i);

            Bb(3 * q + 2, 3 * i)     = -T(2) * N[idx::d12](q, i);
            Bb(3 * q + 2, 3 * i + 1) = -N[idx::d12](q, i);
            Bb(3 * q + 2, 3 * i + 2) = -N[idx::d12](q, i);
        }
    }
    return Bb;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl3p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bs_i (2 x 3): gamma = [w_s1,x, w_s2,y]
    //   row 0 (gamma_x): [ 0   N_i,x  0     ]
    //   row 1 (gamma_y): [ 0   0      N_i,y ]
    Matrix<T> Bs = Matrix<T>::Zero(2 * Q, 3 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bs(2 * q,     3 * i + 1) = N[idx::d1](q, i);
            Bs(2 * q + 1, 3 * i + 2) = N[idx::d2](q, i);
        }
    }
    return Bs;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlinDispl3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlinDispl3p<float>;
#endif

} // namespace pyck
