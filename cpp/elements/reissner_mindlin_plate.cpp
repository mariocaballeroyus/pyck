#include "reissner_mindlin_plate.hpp"
#include "surface.hpp"

namespace pyck
{

template <std::floating_point T>
ReissnerMindlinPlate<T>::ReissnerMindlinPlate(Ptr<PlaneStress2d<T>> material, T k_s)
    : material_(material), k_s_(k_s)
{
    if (!material_) {
        throw std::invalid_argument("ReissnerMindlinPlate: material is null.");
    }

    T E  = material_->youngs_modulus();
    T nu = material_->poisson_ratio();
    T h  = material_->thickness();
    T G  = E / (static_cast<T>(2.0) * (static_cast<T>(1.0) + nu));
    T D  = material_->bending_stiffness();

    // Bending constitutive matrix (Moment-Curvature)
    D_b_ = Matrix<T>::Zero(3, 3);
    D_b_(0, 0) = D;
    D_b_(0, 1) = D * nu;
    D_b_(1, 0) = D * nu;
    D_b_(1, 1) = D;
    D_b_(2, 2) = D * (static_cast<T>(1.0) - nu) / static_cast<T>(2.0);

    // Shear constitutive matrix (Shear force-Shear strain)
    D_s_ = Matrix<T>::Identity(2, 2) * (k_s_ * G * h);
}

template <std::floating_point T>
Matrix<T> ReissnerMindlinPlate<T>::shape_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    const auto& N = shape_derivs[idx::fn];
    const std::size_t Q = N.rows();
    const std::size_t K = N.cols();
    
    // N_mat size (3Q, 3K) for 3 DOFs per node
    Matrix<T> N_mat = Matrix<T>::Zero(3 * Q, 3 * K);
    for (std::size_t q = 0; q < Q; ++q) {
        for (std::size_t i = 0; i < K; ++i) {
            N_mat(3 * q + 0, 3 * i + 0) = N(q, i); // w
            N_mat(3 * q + 1, 3 * i + 1) = N(q, i); // theta_x
            N_mat(3 * q + 2, 3 * i + 2) = N(q, i); // theta_y
        }
    }
    return N_mat;
}

template <std::floating_point T>
Matrix<T> ReissnerMindlinPlate<T>::strain_displacement_matrix(
    const std::vector<Matrix<T>>& shape_derivs) const
{
    // Simplified B-matrix (usually for the first quadrature point)
    const auto& Nu = shape_derivs[idx::u];
    const auto& Nv = shape_derivs[idx::v];
    const auto& N  = shape_derivs[idx::fn];
    const std::size_t K = Nu.cols();
    
    // Total strain vector: [kappa_x, kappa_y, kappa_xy, gamma_x, gamma_y]^T (size 5)
    Matrix<T> B = Matrix<T>::Zero(5, 3 * K);
    for (std::size_t i = 0; i < K; ++i) {
        // Bending (assuming unit Jacobian)
        B(0, 3 * i + 1) = -Nu(0, i);
        B(1, 3 * i + 2) = -Nv(0, i);
        B(2, 3 * i + 1) = -Nv(0, i);
        B(2, 3 * i + 2) = -Nu(0, i);
        
        // Shear
        B(3, 3 * i + 0) = Nu(0, i);
        B(3, 3 * i + 1) = -N(0, i);
        B(4, 3 * i + 0) = Nv(0, i);
        B(4, 3 * i + 2) = -N(0, i);
    }
    return B;
}

template <std::floating_point T>
void ReissnerMindlinPlate<T>::compute_local_stiffness(
    const Patch<T, 2>& patch,
    const ColMatrix<T, 2>& q_points,
    const Vector<T>& q_weights,
    Index span,
    Matrix<T>& stiffness) const
{
    auto [shape_fns, jac] = patch.eval_shape_functions(q_points, span, 1);
    Vector<T> dV = q_weights.cwiseProduct(jac);

    const Index Q = q_points.rows();
    const Index K = shape_fns[0].cols();
    stiffness.setZero(3 * K, 3 * K);

    // Get active control points to compute physical derivatives
    const auto& s_patch = dynamic_cast<const SurfacePatch<T>&>(patch);
    auto act_pts = s_patch.active_control_pts(s_patch.decode_span(span));

    for (Index q = 0; q < Q; ++q)
    {
        if (std::abs(jac(q)) < T(1e-14)) continue;

        const auto& N = shape_fns[idx::fn];
        const auto& Nu = shape_fns[idx::u];
        const auto& Nv = shape_fns[idx::v];

        // Compute tangent vectors
        Eigen::Matrix<T, 1, 3> a1 = Nu.row(q) * act_pts;
        Eigen::Matrix<T, 1, 3> a2 = Nv.row(q) * act_pts;

        // In-plane Jacobian matrix (u,v) -> (x,y)
        Eigen::Matrix<T, 2, 2> J;
        J << a1(0), a1(1),
             a2(0), a2(1);
        
        T detJ = J.determinant();
        if (std::abs(detJ) < T(1e-14)) continue;
        Eigen::Matrix<T, 2, 2> invJ = J.inverse();

        Matrix<T> B_b = Matrix<T>::Zero(3, 3 * K);
        Matrix<T> B_s = Matrix<T>::Zero(2, 3 * K);

        for (Index i = 0; i < K; ++i)
        {
            // Physical derivatives
            T dNdx = invJ(0, 0) * Nu(q, i) + invJ(0, 1) * Nv(q, i);
            T dNdy = invJ(1, 0) * Nu(q, i) + invJ(1, 1) * Nv(q, i);

            // Bending: kappa = { kappa_x, kappa_y, kappa_xy }
            //   kappa_x  = -theta_x,x
            //   kappa_y  = -theta_y,y
            //   kappa_xy = -(theta_x,y + theta_y,x)
            B_b(0, 3 * i + 1) = -dNdx;
            B_b(1, 3 * i + 2) = -dNdy;
            B_b(2, 3 * i + 1) = -dNdy;
            B_b(2, 3 * i + 2) = -dNdx;

            // Transverse shear: gamma = { gamma_x, gamma_y }
            //   gamma_x = w,x - theta_x
            //   gamma_y = w,y - theta_y
            B_s(0, 3 * i)     = dNdx;
            B_s(0, 3 * i + 1) = -N(q, i);
            B_s(1, 3 * i)     = dNdy;
            B_s(1, 3 * i + 2) = -N(q, i);
        }

        stiffness.noalias() += B_b.transpose() * (D_b_ * dV(q)) * B_b;
        stiffness.noalias() += B_s.transpose() * (D_s_ * dV(q)) * B_s;
    }
}

// === Template Instantiations ========================================================

template class ReissnerMindlinPlate<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ReissnerMindlinPlate<float>;
#endif

} // namespace pyck
