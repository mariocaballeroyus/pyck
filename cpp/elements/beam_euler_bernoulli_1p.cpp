#include "beam_euler_bernoulli_1p.hpp"
#include "patch.hpp"
#include "christoffels.hpp"

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T>
BeamEulerBernoulli1p<T>::BeamEulerBernoulli1p(Ptr<UniaxialStress1d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("BeamEulerBernoulli1p: "
                                    "material is null.");
    }
}

// === Matrix Operators =============================================================??

template <std::floating_point T> 
Matrix<T>
BeamEulerBernoulli1p<T>::strain_matrix(const Patch<T, 1>& /*patch*/,
                                       const BasisDerivs<T, 1>& basis,
                                       const LocalFrame<T, 1>& local) const
{
    auto chr = eval_christoffel(local);

    const Index Q = basis.N_u.rows();
    const Index n = basis.N_u.cols();
    Matrix<T> B(Q, n);

    for (Index q = 0; q < Q; ++q)
    {
        // Bending
        // B_b = [ -N_{i|11} ]
        B.row(q) = -(basis.N_uu.row(q) - chr.G1_11(q) * basis.N_u.row(q));

        // B_s = 0 (normality assumption)
    }
    return B;
}

template <std::floating_point T>
Matrix<T> 
BeamEulerBernoulli1p<T>::constitutive_matrix(const LocalFrame<T, 1>& local,
                                             Index q) const
{
    const T gi = local.g_inv_11(q);
    Matrix<T> D(1, 1);

    // D_b = [ EI (g^{11})^2 ]
    D(0, 0) = material_->bending_stiffness() * gi * gi;
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T> 
Matrix<T>
BeamEulerBernoulli1p<T>::displacement_shape_matrix(const Patch<T, 1>& /*patch*/,
                                                   const BasisDerivs<T, 1>& basis,
                                                   const LocalFrame<T, 1>& /*local*/) const
{
    // N_w = [ N_i ]
    return basis.N;
}

template <std::floating_point T> 
Matrix<T>
BeamEulerBernoulli1p<T>::rotation_shape_matrix(const Patch<T, 1>& /*patch*/,
                                               const BasisDerivs<T, 1>& basis,
                                               const LocalFrame<T, 1>& /*local*/) const
{
    // N_rot = [ -N_{i|1} ]
    return -basis.N_u;
}

// === Template Instantiations ========================================================

template class BeamEulerBernoulli1p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class BeamEulerBernoulli1p<float>;
#endif

} // namespace pyck
