#include "shell_kirchhoff_love_3p.hpp"
#include "patch.hpp"
#include "primitives_intrinsic.hpp"

namespace pyck
{

template <std::floating_point T>
ShellKirchhoffLove3p<T>::ShellKirchhoffLove3p(Ptr<PlaneStress2d<T>> material)
    : material_(material)
{
    if (!material_) {
        throw std::invalid_argument("ShellKirchhoffLove3p: material is null.");
    }
}

// === Matrix Operators ===============================================================

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::strain_matrix(const ElementValues<T, 2>& ev) const
{
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    const ColMatrix<T, 3>& a_3 = ev.n;

    Matrix<T>& B = this->B_workspace_;
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    B.setZero(6 * Q, 3 * N);

    for (Index q = 0; q < Q; ++q)
    {
        auto slab1 = ev.results_[1].col(q);
        auto slab2 = ev.results_[2].col(q);

        const Vec3 A1 = ev.a(0).row(q).transpose();   // covariant tangent A_1
        const Vec3 A2 = ev.a(1).row(q).transpose();   // covariant tangent A_2
        const Vec3 A3 = a_3.row(q).transpose();        // unit normal A_3

        const T inv_sqrtA = T(1) / ev.jac(q);

        // Both strain measures are first variations of the surface fundamental forms,
        // linearized about the undeformed configuration; reference (undeformed) quantities
        // are capital — base vectors A_α = X_{,α}, second derivatives A_{,αβ} = X_{,αβ},
        // unit normal A_3. ( ),r is the variation w.r.t. displacement DOF r:
        //
        //   Membrane:  ε_{αβ} = ½ A_{αβ},r = ½(A_{α,r}·A_β + A_{β,r}·A_α),   A_{α,r} = u_{,α}.
        //   Bending:   κ_{αβ} = −B_{αβ},r = −(A_{,αβ},r·A_3 + A_{,αβ}·A_{3,r}),
        //
        // i.e. the negated variation of the curvature B_{αβ} = A_{,αβ}·A_3. Here
        // A_{,αβ},r = u_{,αβ} (X_{,αβ} is linear in the DOFs) is the displacement second
        // derivative, and A_{3,r} = δA_3 is the normal variation,
        //     A_{3,r} = (1/√A)[ δ(A_1×A_2) − (A_3·δ(A_1×A_2)) A_3 ],
        //             δ(A_1×A_2) = u_{,1}×A_2 + A_1×u_{,2}.
        // A_{3,r} is never formed; the product A_{,αβ}·A_{3,r} is assembled directly from
        // the basis-independent factors below. (Geometric tangent for the nonlinear solver,
        // evaluated at the current configuration — lowercase a — and not assembled here:
        // b_{αβ},rs = a_{α,β},r·a_3,s + a_{α,β},s·a_3,r + a_{α,β}·a_3,rs, with a_{α,β},rs = 0.)

        // Factors of A_3·δ(A_1×A_2), the unit-normal projection subtracted inside A_{3,r}:
        const Vec3 c1 = A2.cross(A3);   // A_2×A_3  (the u_{,1} = N^i_{,1} e_k contribution)
        const Vec3 c2 = A3.cross(A1);   // A_3×A_1  (the u_{,2} = N^i_{,2} e_k contribution)

        const Vec3 A11 = ev.a_d1(0, 0).row(q).transpose();   // A_{,11} = X_{,11}
        const Vec3 A22 = ev.a_d1(1, 1).row(q).transpose();   // A_{,22} = X_{,22}
        const Vec3 A12 = ev.a_d1(0, 1).row(q).transpose();   // A_{,12} = X_{,12}

        // Factors of A_{,αβ}·δ(A_1×A_2), the reference curvature acting on the normal tilt:
        const Vec3 p11 = A2.cross(A11), r11 = A11.cross(A1);  // A_2×A_{,11}, A_{,11}×A_1
        const Vec3 p22 = A2.cross(A22), r22 = A22.cross(A1);  // A_2×A_{,22}, A_{,22}×A_1
        const Vec3 p12 = A2.cross(A12), r12 = A12.cross(A1);  // A_2×A_{,12}, A_{,12}×A_1

        const T B11 = ev.b(0, 0)(q), B22 = ev.b(1, 1)(q), B12 = ev.b(0, 1)(q);  // B_{αβ} = A_{,αβ}·A_3

        for (Index i = 0; i < N; ++i)
        {
            const T Nu  = slab1(i * 2 + 0);   // N^i_{,1}    ┐ A_{α,r}   = N^i_{,α} e_k  (tangent variation)
            const T Nv  = slab1(i * 2 + 1);   // N^i_{,2}    ┘
            const T Nuu = slab2(i * 3 + 0);   // N^i_{,11}   ┐
            const T Nvv = slab2(i * 3 + 1);   // N^i_{,22}   ┼ A_{,αβ},r = N^i_{,αβ} e_k (2nd-deriv variation)
            const T Nuv = slab2(i * 3 + 2);   // N^i_{,12}   ┘

            // Column block of DOF r = (i,k); ( )(k) picks the k-th Cartesian component.
            for (Index k = 0; k < 3; ++k)
            {
                // Membrane ε_{αβ} = ½ A_{αβ},r = ½(A_{α,r}·A_β + A_{β,r}·A_α):
                B(6*q,     3*i + k) = Nu * A1(k);                // ε_11  = A_{1,r}·A_1
                B(6*q + 1, 3*i + k) = Nv * A2(k);                // ε_22  = A_{2,r}·A_2
                B(6*q + 2, 3*i + k) = Nu * A2(k) + Nv * A1(k);   // 2ε_12 = A_{1,r}·A_2 + A_{2,r}·A_1

                // Bending κ_{αβ} = −(A_{,αβ},r·A_3 + A_{,αβ}·A_{3,r}). Per row: the first
                // summand N_{αβ}·A3(k) is A_{,αβ},r·A_3; the bracket is A_{,αβ}·A_{3,r} =
                // (1/√A)(A_{,αβ}·δ(A_1×A_2) − B_{αβ}·proj), with proj = A_3·δ(A_1×A_2).
                const T proj = Nu * c1(k) + Nv * c2(k);                          // proj = A_3·δ(A_1×A_2)
                B(6*q + 3, 3*i + k) = -( Nuu * A3(k)                             //   A_{,11},r·A_3
                    + inv_sqrtA * ( Nu * p11(k) + Nv * r11(k) - B11 * proj ) );  // + A_{,11}·A_{3,r}
                B(6*q + 4, 3*i + k) = -( Nvv * A3(k)                             //   A_{,22},r·A_3
                    + inv_sqrtA * ( Nu * p22(k) + Nv * r22(k) - B22 * proj ) );  // + A_{,22}·A_{3,r}
                B(6*q + 5, 3*i + k) = -T(2) * ( Nuv * A3(k)                      //   A_{,12},r·A_3
                    + inv_sqrtA * ( Nu * p12(k) + Nv * r12(k) - B12 * proj ) );  // + A_{,12}·A_{3,r}   (×2 ⇒ 2κ_12)
            }
        }
    }
}

template <std::floating_point T>
ConstitutiveMatrix<T>
ShellKirchhoffLove3p<T>::constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const
{
    // D = [ D_m  0   ]
    //     [ 0    D_b ]
    const Eigen::Matrix<T, 3, 1> g_inv_q = g_inv_voigt(ev, q);

    ConstitutiveMatrix<T> D = ConstitutiveMatrix<T>::Zero(6, 6);
    D.template block<3, 3>(0, 0) = material_->membrane_voigt(g_inv_q);   // t · C
    D.template block<3, 3>(3, 3) = material_->bending_voigt (g_inv_q);   // (t³/12) · C
    return D;
}

// === Shape Matrices =================================================================

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::displacement_shape_matrix(const ElementValues<T, 2>& ev) const
{
    // Physical displacement u = (u_x, u_y, u_z): the three Cartesian DOFs.
    const Index Q = ev.results_[0].cols();
    const Index N = ev.results_[0].rows();
    Matrix<T>& U = this->N_w_workspace_;
    U.setZero(3 * Q, 3 * N);
    for (Index q = 0; q < Q; ++q) {
        auto slab0 = ev.results_[0].col(q);
        for (Index i = 0; i < N; ++i) {
            const T Ni = slab0(i);
            U(3 * q + 0, 3 * i + 0) = Ni;   // u_x
            U(3 * q + 1, 3 * i + 1) = Ni;   // u_y
            U(3 * q + 2, 3 * i + 2) = Ni;   // u_z
        }
    }
}

template <std::floating_point T>
void
ShellKirchhoffLove3p<T>::rotation_shape_matrix(const ElementValues<T, 2>&) const
{
    throw std::runtime_error("ShellKirchhoffLove3p::rotation_shape_matrix: "
                             "not implemented.");
}

// === Template Instantiations ========================================================

template class ShellKirchhoffLove3p<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ShellKirchhoffLove3p<float>;
#endif

} // namespace pyck
