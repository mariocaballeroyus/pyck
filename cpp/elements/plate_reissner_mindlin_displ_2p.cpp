#include "plate_reissner_mindlin_displ_2p.hpp"

namespace pyck
{

template <std::floating_point T>
PlateReissnerMindlinDispl2p<T>::PlateReissnerMindlinDispl2p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("PlateReissnerMindlinDispl2p: material is null.");
    }
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::displacement_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    // Nw_i = [ N_i - (Kb/Ks)(N_i,xx + N_i,yy)   0 ]
    Matrix<T> Nw = Matrix<T>::Zero(Q, 2 * n);
    for (Index i = 0; i < n; ++i) {
        Nw.col(2 * i) = N[idx::val].col(i)
                      - ratio * (N[idx::d11].col(i) + N[idx::d22].col(i));
    }
    return Nw;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::rotation_shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 2 * n);

    // Nphi_i = [ -N_i,x   N_i,y
    //            -N_i,y  -N_i,x ]
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Nphi(2 * q,     2 * i)     = -N[idx::d1](q, i);
            Nphi(2 * q,     2 * i + 1) =  N[idx::d2](q, i);
            Nphi(2 * q + 1, 2 * i)     = -N[idx::d2](q, i);
            Nphi(2 * q + 1, 2 * i + 1) = -N[idx::d1](q, i);
        }
    }
    return Nphi;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::bending_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();

    // Bb_i (3 x 2): kappa = L phi
    //   row 0 (kappa_x):    [ -N_i,xx              N_i,xy             ]
    //   row 1 (kappa_y):    [ -N_i,yy             -N_i,xy             ]
    //   row 2 (2 kappa_xy): [ -2 N_i,xy            N_i,yy - N_i,xx    ]
    Matrix<T> Bb = Matrix<T>::Zero(3 * Q, 2 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bb(3 * q,     2 * i)     = -N[idx::d11](q, i);
            Bb(3 * q,     2 * i + 1) =  N[idx::d12](q, i);

            Bb(3 * q + 1, 2 * i)     = -N[idx::d22](q, i);
            Bb(3 * q + 1, 2 * i + 1) = -N[idx::d12](q, i);

            Bb(3 * q + 2, 2 * i)     = -T(2) * N[idx::d12](q, i);
            Bb(3 * q + 2, 2 * i + 1) =  N[idx::d22](q, i) - N[idx::d11](q, i);
        }
    }
    return Bb;
}

template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::shear_strain_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    const T ratio = material_->bending_stiffness() / material_->shear_stiffness();

    // Bs_i (2 x 2): gamma = -(Kb/Ks) grad(Lap w_b) + curl psi
    //   row 0 (gamma_x): [ -(Kb/Ks)(N_i,xxx + N_i,xyy)   N_i,y ]
    //   row 1 (gamma_y): [ -(Kb/Ks)(N_i,xxy + N_i,yyy)  -N_i,x ]
    Matrix<T> Bs = Matrix<T>::Zero(2 * Q, 2 * n);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            Bs(2 * q,     2 * i)     = -ratio * (N[idx::d111](q, i)
                                               + N[idx::d122](q, i));
            Bs(2 * q,     2 * i + 1) =  N[idx::d2](q, i);

            Bs(2 * q + 1, 2 * i)     = -ratio * (N[idx::d112](q, i)
                                               + N[idx::d222](q, i));
            Bs(2 * q + 1, 2 * i + 1) = -N[idx::d1](q, i);
        }
    }
    return Bs;
}

// Direct moment recovery from the primary DOFs (w_b, psi):
//   m = Db * kappa,  kappa = L(-grad(w_b) + curl(psi))
// Per node i with DOFs [w_b, psi], the kappa stencil is the same 3x2 block
// used in bending_strain_matrix:
//   kappa_i = [ -N_i,xx              N_i,xy
//              -N_i,yy             -N_i,xy
//              -2 N_i,xy            N_i,yy - N_i,xx ]
// so the moment shape block is N_m_i = Db * kappa_i (3 x 2). No projection
// or auxiliary problem: m_h is a pointwise linear functional of (w_b, psi)
// through Db (constant) and second derivatives of the basis.
template <std::floating_point T>
Matrix<T> PlateReissnerMindlinDispl2p<T>::moment_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index Q = N[idx::val].rows();
    const Index n = N[idx::val].cols();
    const Matrix<T> Db = material_->bending_matrix();
    Matrix<T> Nm = Matrix<T>::Zero(3 * Q, 2 * n);

    Matrix<T> kappa_i(3, 2);
    for (Index q = 0; q < Q; ++q) {
        for (Index i = 0; i < n; ++i) {
            kappa_i(0, 0) = -N[idx::d11](q, i);
            kappa_i(0, 1) =  N[idx::d12](q, i);
            kappa_i(1, 0) = -N[idx::d22](q, i);
            kappa_i(1, 1) = -N[idx::d12](q, i);
            kappa_i(2, 0) = -T(2) * N[idx::d12](q, i);
            kappa_i(2, 1) =  N[idx::d22](q, i) - N[idx::d11](q, i);

            Nm.block(3 * q, 2 * i, 3, 2).noalias() = Db * kappa_i;
        }
    }
    return Nm;
}

// === Template Instantiations ========================================================

template class PlateReissnerMindlinDispl2p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PlateReissnerMindlinDispl2p<float>;
#endif

} // namespace pyck
