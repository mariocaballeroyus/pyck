#include "kirchhoff_love_plate.hpp"

namespace pyck
{

template <std::floating_point T>
KirchhoffLovePlate<T>::KirchhoffLovePlate(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("KirchhoffLovePlate: material is null.");
    }

    T nu = material_->poisson_ratio();
    T D = material_->bending_stiffness();

    // Bending constitutive matrix
    //   D_b = D * [ 1    ν     0        ]
    //             [ ν    1     0        ]
    //             [ 0    0   (1-ν)/2   ]
    D_b_ = Matrix<T>::Zero(3, 3);
    D_b_(0, 0) = D;
    D_b_(0, 1) = D * nu;
    D_b_(1, 0) = D * nu;
    D_b_(1, 1) = D;
    D_b_(2, 2) = D * (T(1) - nu) / T(2);
}

template <std::floating_point T>
Matrix<T> KirchhoffLovePlate<T>::shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // For 1 DOF/node (transverse displacement), N is just the
    // tensor-product basis function values: shape (Q, K).
    return shape_derivs[0];
}

template <std::floating_point T>
Matrix<T> KirchhoffLovePlate<T>::strain_displacement_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // B matrix from the second covariant derivatives (first quad point).
    //   κ = { κ_uu, κ_vv, 2κ_uv } = { N_{;uu}, N_{;vv}, 2 N_{;uv} } w
    const Index K = shape_derivs[0].cols();
    Matrix<T> B(3, K);
    B.row(0) = shape_derivs[3].row(0);           // N_{;uu}
    B.row(1) = shape_derivs[5].row(0);           // N_{;vv}
    B.row(2) = T(2) * shape_derivs[4].row(0);    // 2 N_{;uv}
    return B;
}

template <std::floating_point T>
void KirchhoffLovePlate<T>::compute_local_stiffness(
    const Patch<T, 2>& patch,
    const ColMatrix<T, 2>& q_points,
    const Vector<T>& q_weights,
    Index span,
    Matrix<T>& stiffness) const
{
    // Second-order covariant derivatives needed for curvature.
    auto [shape_fns, jac] = patch.eval_shape_functions(q_points, span, 2);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    const Index Q = q_points.rows();
    const Index K = shape_fns[0].cols();
    stiffness.setZero(K, K);

    for (Index q = 0; q < Q; ++q)
    {
        // B(q) = [ N_{;uu}(q) ]   (3 × K)
        //        [ N_{;vv}(q) ]
        //        [ 2 N_{;uv}(q) ]
        Matrix<T> B(3, K);
        B.row(0) = shape_fns[3].row(q);
        B.row(1) = shape_fns[5].row(q);
        B.row(2) = T(2) * shape_fns[4].row(q);

        stiffness.noalias() += B.transpose() * (D_b_ * dV(q)) * B;
    }
}

// === Template Instantiations ========================================================

template class KirchhoffLovePlate<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class KirchhoffLovePlate<float>;
#endif

} // namespace pyck
