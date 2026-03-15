#include "kirchhoff_love_plate.hpp"

namespace pyck
{

template <std::floating_point T>
KirchhoffLovePlate<T>::KirchhoffLovePlate(T youngs_modulus,
                                          T poisson_ratio,
                                          T thickness)
    : E_(youngs_modulus), nu_(poisson_ratio), h_(thickness),
      Kb_(youngs_modulus * thickness * thickness * thickness
         / (T(12) * (T(1) - poisson_ratio * poisson_ratio)))
{
    if (E_ <= 0) {
        throw std::invalid_argument("KirchhoffLovePlate: "
                                    "Young's modulus must be positive.");
    }
    if (nu_ < 0 || nu_ >= T(0.5)) {
        throw std::invalid_argument("KirchhoffLovePlate: "
                                    "Poisson's ratio must be in [0, 0.5).");
    }
    if (h_ <= 0) {
        throw std::invalid_argument("KirchhoffLovePlate: "
                                    "thickness must be positive.");
    }

    // Bending constitutive matrix
    //   Kb_mat = Kb * [ 1    ν     0        ]
    //                 [ ν    1     0        ]
    //                 [ 0    0   (1-ν)/2   ]
    Kb_mat_ = Matrix<T>::Zero(3, 3);
    Kb_mat_(0, 0) = Kb_;
    Kb_mat_(0, 1) = Kb_ * nu_;
    Kb_mat_(1, 0) = Kb_ * nu_;
    Kb_mat_(1, 1) = Kb_;
    Kb_mat_(2, 2) = Kb_ * (T(1) - nu_) / T(2);
}

template <std::floating_point T>
Matrix<T> KirchhoffLovePlate<T>::shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    return shape_derivs[0];
}

template <std::floating_point T>
Matrix<T> KirchhoffLovePlate<T>::strain_displacement_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs;
    const Index K = N[idx::fn].cols();

    Matrix<T> B(3, K);
    B.row(0) = N[idx::uu].row(0);
    B.row(1) = N[idx::vv].row(0);
    B.row(2) = T(2) * N[idx::uv].row(0);
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

    const auto& N = shape_fns;
    for (Index q = 0; q < Q; ++q)
    {
        Matrix<T> B(3, K);
        B.row(0) = N[idx::uu].row(q);
        B.row(1) = N[idx::vv].row(q);
        B.row(2) = T(2) * N[idx::uv].row(q);

        stiffness.noalias() += B.transpose() * (Kb_mat_ * dV(q)) * B;
    }
}

// === Template Instantiations ========================================================

template class KirchhoffLovePlate<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class KirchhoffLovePlate<float>;
#endif

} // namespace pyck
