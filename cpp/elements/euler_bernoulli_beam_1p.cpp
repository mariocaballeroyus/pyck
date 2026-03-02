#include "euler_bernoulli_beam_1p.hpp"

namespace pyck
{

template <std::floating_point T>
void EulerBernoulliBeam1P<T>::compute_local_stiffness(const Patch<T, 1>& patch,
                                                      const ColMatrix<T, 1>& q_points,
                                                      const Vector<T>& q_weights,
                                                      Matrix<T>& stiffness) const
{
    // Extract shape function values and derivatives (up to 2nd order)
    std::vector<Matrix<T>> shape_fns = patch.eval_shape_functions(q_points, 2);

    const int num_qp = q_points.rows();
    const int num_basis = patch.basis(0).num_basis();

    // Initialize local stiffness matrix
    stiffness.setZero(num_basis, num_basis);

    Vector<T> jac = patch.eval_jacobian(q_points);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    const Matrix<T>& d2N = shape_fns[2]; // Second parametric derivatives

    for (int i = 0; i < num_qp; ++i)
    {
        // d2N already contains the physical second derivatives d^2N/dx^2 
        // because Patch::eval_shape_functions applies the covariant derivative mapping.
        RowVector<T> Bb_i = d2N.row(i);

        // Ke += B^T * EI * B * dV
        stiffness.noalias() += Kb_ * (Bb_i.transpose() * Bb_i) * dV(i);
    }
}

template <std::floating_point T>
void EulerBernoulliBeam1P<T>::compute_local_load(const Patch<T, 1>& patch,
                                                 const ColMatrix<T, 1>& q_points,
                                                 const Vector<T>& q_weights,
                                                 const Vector<T>& load_values,
                                                 Vector<T>& load) const
{
    // Extract shape function values (0th order)
    std::vector<Matrix<T>> shape_fns = patch.eval_shape_functions(q_points, 0);

    const int num_basis = patch.basis(0).num_basis();

    load.setZero(num_basis);

    Vector<T> jac = patch.eval_jacobian(q_points);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    const Matrix<T>& N = shape_fns[0];

    // F += N^T * diag(load_values) * dV
    Vector<T> W = load_values.cwiseProduct(dV);
    load.noalias() = N.transpose() * W;
}

// === Template Instantiations ========================================================

template class EulerBernoulliBeam1P<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class EulerBernoulliBeam1P<float>;
#endif

} // namespace pyck
