#include "euler_bernoulli_beam_1p.hpp"

namespace pyck
{

template <std::floating_point T>
void EulerBernoulliBeam1P<T>::compute_local_stiffness(const Patch<T, 1>& patch,
                                                      const QuadratureRule<T, 1>& quadrature,
                                                      Matrix<T>& stiffness) const
{
    // Extract shape function values and derivatives (up to 2nd order)
    std::vector<Matrix<T>> shape_fns = patch.eval_shape_functions(quadrature.points(), 2);

    const int num_qp = quadrature.points().rows();
    const int num_basis = patch.basis(0).num_basis();

    // Initialize local stiffness matrix
    stiffness.setZero(num_basis, num_basis);

    Vector<T> jac = patch.eval_jacobian(quadrature.points());
    Vector<T> dV = quadrature.weights().cwiseProduct(jac);

    const Matrix<T>& Bb = shape_fns[2]; // Bending strain-displacement matrix

    for (int i = 0; i < num_qp; ++i)
    {
        // (1, num_basis) bending strain at quadrature point i
        RowVector<T> Bb_i = Bb.row(i);

        // Ke += B^T * EI * B * dV
        stiffness.noalias() += Kb_ * (Bb_i.transpose() * Bb_i) * dV(i);
    }
}

template <std::floating_point T>
void EulerBernoulliBeam1P<T>::compute_local_load(const Patch<T, 1>& patch,
                                                 const QuadratureRule<T, 1>& quadrature,
                                                 const std::function<T(const ColMatrix<T, 1>&)>& load_fn,
                                                 Vector<T>& load) const
{
    // Extract shape function values (0th order)
    std::vector<Matrix<T>> shape_fns = patch.eval_shape_functions(quadrature.points(), 0);

    const int num_qp = quadrature.points().rows();
    const int num_basis = patch.basis(0).num_basis();

    load.setZero(num_basis);

    Vector<T> jac = patch.eval_jacobian(quadrature.points());
    Vector<T> dV = quadrature.weights().cwiseProduct(jac);

    const Matrix<T>& N = shape_fns[0]; 
    std::vector<ColMatrix<T, 3>> phys_pts = patch.eval_geometry(quadrature.points(), 0);
    const ColMatrix<T, 3>& X = phys_pts[0];

    for (int i = 0; i < num_qp; ++i)
    {
        // Extract the first coordinate of the 3D geometry point matching the load function
        ColMatrix<T, 1> x_pt = X.row(i).head(1);
        T load_val = load_fn(x_pt);
        // F += N^T * p * dV
        load.noalias() += N.row(i).transpose() * load_val * dV(i);
    }
}

// === Template Instantiations ========================================================

template class EulerBernoulliBeam1P<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class EulerBernoulliBeam1P<float>;
#endif

} // namespace pyck
