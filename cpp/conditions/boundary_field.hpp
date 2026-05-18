#ifndef PYCK_BOUNDARY_FIELD_HPP
#define PYCK_BOUNDARY_FIELD_HPP

#include <concepts>

#include "../elements/element.hpp"
#include "../geometry/patch_boundary.hpp"
#include "../geometry/intrinsic_geometry.hpp"
#include "../geometry/extrinsic_geometry.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Polymorphic boundary-field operator.
 */
template <std::floating_point T>
class BoundaryField
{
public:

    /**
     * @brief Default destructor.
     */
    virtual ~BoundaryField() = default;

    /**
     * @brief Evaluate the boundary field.
     * 
     * @param element The element to evaluate the boundary field on.
     * @param boundary The boundary of the patch.
     * @param boundary_span The span of the boundary.
     * @param boundary_basis The basis derivatives of the boundary.
     * @param boundary_local The local frame of the boundary.
     * @param parent_flat_span The flat span of the parent.
     * @param parent_basis The basis derivatives of the parent.
     * @param parent_ig The intrinsic geometry of the parent.
     * @return The boundary field evaluated at the quadrature points.
     */
    virtual Matrix<T> evaluate(const Element<T, 2>& element,
                               const PatchBoundary<T, 2>& boundary,
                               Index boundary_span,
                               const BasisDerivs<T, 1>& boundary_basis,
                               const IntrinsicGeometry<T, 1>& boundary_local,
                               Index parent_flat_span,
                               const BasisDerivs<T, 2>& parent_basis,
                               const IntrinsicGeometry<T, 2>& parent_ig) const = 0;
};

namespace detail
{

/// Covariant components of a 3D in-surface vector v in the parent surface:
///   v_α = v · a_α.
template <std::floating_point T>
inline std::pair<Vector<T>, Vector<T>>
covariant_components(const ColMatrix<T, 3>& v,
                     const IntrinsicGeometry<T, 2>& local)
{
    const Index Q = v.rows();
    Vector<T> v_cov_1(Q), v_cov_2(Q);
    for (Index q = 0; q < Q; ++q) {
        v_cov_1(q) = v.row(q).dot(local.a[0].row(q));
        v_cov_2(q) = v.row(q).dot(local.a[1].row(q));
    }
    return {std::move(v_cov_1), std::move(v_cov_2)};
}

/// Contravariant components of a 3D in-surface vector v in the parent surface:
///   v^α = g^{αβ} (v · a_β).
/// Pairs naturally with covariant strains/derivatives (θ_α, N_{,α}, N_{|αβ}).
template <std::floating_point T>
inline std::pair<Vector<T>, Vector<T>>
contravariant_components(const ColMatrix<T, 3>& v,
                         const IntrinsicGeometry<T, 2>& local)
{
    auto [v_cov_1, v_cov_2] = covariant_components(v, local);
    const Index Q = v.rows();
    Vector<T> v_up_1(Q), v_up_2(Q);
    for (Index q = 0; q < Q; ++q) {
        const T gi11 = local.g_inv[0][0](q);
        const T gi12 = local.g_inv[0][1](q);
        const T gi22 = local.g_inv[1][1](q);
        v_up_1(q) = gi11 * v_cov_1(q) + gi12 * v_cov_2(q);
        v_up_2(q) = gi12 * v_cov_1(q) + gi22 * v_cov_2(q);
    }
    return {std::move(v_up_1), std::move(v_up_2)};
}

/// In-surface unit tangent perpendicular to the outward normal n, oriented
/// CCW in the surface:  s = a_3 × n.  Both a_3 and n are unit and mutually
/// perpendicular, so |s| = 1.
template <std::floating_point T>
inline ColMatrix<T, 3>
surface_tangent(const ColMatrix<T, 3>& n,
                const ColMatrix<T, 3>& a_3)
{
    const Index Q = n.rows();
    ColMatrix<T, 3> s(Q, 3);
    for (Index q = 0; q < Q; ++q) {
        const Eigen::Matrix<T, 3, 1> a3q = a_3.row(q).transpose();
        const Eigen::Matrix<T, 3, 1> nq  = n.row(q).transpose();
        s.row(q) = a3q.cross(nq).transpose();
    }
    return s;
}

} // namespace detail

/**
 * @brief Basis-DOF value field: extracts the spline value of a single DOF
 *        slot within each node block, ignoring all others.
 *
 * Used as the `C^0` building block for inter-patch coupling: pinning a
 * primary DOF directly (e.g. `w_b` at index 0 or `psi` at index 1 on
 * `PlateReissnerMindlinDispl2p`) sidesteps the linear-combination
 * identities baked into the element's `displacement_shape_matrix`.
 */
template <std::floating_point T>
class BasisValue : public BoundaryField<T>
{
public:
    explicit BasisValue(std::size_t dof_index = 0) : dof_index_(dof_index) {}

    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& /*boundary*/,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& /*boundary_local*/,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& /*parent_ig*/) const override
    {
        const Index ndof = static_cast<Index>(element.num_node_dofs());
        const auto N = parent_basis.N();
        const Index Q = N.rows();
        const Index n = N.cols();

        Matrix<T> C = Matrix<T>::Zero(Q, n * ndof);
        const Index slot = static_cast<Index>(dof_index_);
        for (Index i = 0; i < n; ++i) {
            C.col(i * ndof + slot) = N.col(i);
        }
        return C;
    }

private:
    std::size_t dof_index_;
};

/**
 * @brief Basis-DOF normal-slope field: `n · ∇N` of a single DOF slot.
 *
 *        Computed intrinsically as `n^α N_{,α}` where n^α are the contravariant
 *        components of the boundary outward in-surface normal (built from the
 *        parent patch's inverse metric).  Reduces to `n_x N_x + n_y N_y` on
 *        flat axis-aligned plates.
 *
 *        `C^1` building block. With opposing outward normals across the
 *        interface this field carries `sign_b = -1` (one application of
 *        `n` flips sign).
 */
template <std::floating_point T>
class BasisNormalSlope : public BoundaryField<T>
{
public:
    explicit BasisNormalSlope(std::size_t dof_index = 0) : dof_index_(dof_index) {}

    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& boundary_local,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        const Index ndof = static_cast<Index>(element.num_node_dofs());
        const auto N_u = parent_basis.N_d1(0);
        const auto N_v = parent_basis.N_d1(1);
        const Index Q = N_u.rows();
        const Index n = N_u.cols();

        const ColMatrix<T, 3> normal =
            boundary.eval_outward_normal(boundary_local, parent_ig);
        const auto [n_up_1, n_up_2] =
            detail::contravariant_components(normal, parent_ig);

        Matrix<T> C = Matrix<T>::Zero(Q, n * ndof);
        const Index slot = static_cast<Index>(dof_index_);
        for (Index q = 0; q < Q; ++q) {
            for (Index i = 0; i < n; ++i) {
                C(q, i * ndof + slot) =
                    n_up_1(q) * N_u(q, i) + n_up_2(q) * N_v(q, i);
            }
        }
        return C;
    }

private:
    std::size_t dof_index_;
};

/**
 * @brief Basis-DOF normal-curvature field: `n^α n^β N_{|αβ}` of a single
 *        DOF slot, where N_{|αβ} = N_{,αβ} − Γ^γ_{αβ} N_{,γ} is the covariant
 *        Hessian and n^α the contravariant normal components.
 *
 *        Reduces to `n_x² N,xx + 2 n_x n_y N,xy + n_y² N,yy` on flat
 *        axis-aligned plates where Γ ≡ 0 and a_α coincide with (ê_x, ê_y).
 *
 *        `C^2` building block. With opposing outward normals across the
 *        interface this field carries `sign_b = +1` (two applications of
 *        `n` cancel).
 */
template <std::floating_point T>
class BasisNormalCurvature : public BoundaryField<T>
{
public:
    explicit BasisNormalCurvature(std::size_t dof_index = 0) : dof_index_(dof_index) {}

    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& boundary_local,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        const Index ndof = static_cast<Index>(element.num_node_dofs());
        const auto N_u  = parent_basis.N_d1(0);
        const auto N_v  = parent_basis.N_d1(1);
        const auto N_uu = parent_basis.N_d2(0, 0);
        const auto N_uv = parent_basis.N_d2(0, 1);
        const auto N_vv = parent_basis.N_d2(1, 1);
        const Index Q = N_uu.rows();
        const Index n = N_uu.cols();

        const ColMatrix<T, 3> normal =
            boundary.eval_outward_normal(boundary_local, parent_ig);
        const auto [n_up_1, n_up_2] =
            detail::contravariant_components(normal, parent_ig);

        Matrix<T> C = Matrix<T>::Zero(Q, n * ndof);
        const Index slot = static_cast<Index>(dof_index_);
        for (Index q = 0; q < Q; ++q) {
            const T G1_11 = parent_ig.chr.Gamma[0][0][0](q), G1_12 = parent_ig.chr.Gamma[0][0][1](q), G1_22 = parent_ig.chr.Gamma[0][1][1](q);
            const T G2_11 = parent_ig.chr.Gamma[1][0][0](q), G2_12 = parent_ig.chr.Gamma[1][0][1](q), G2_22 = parent_ig.chr.Gamma[1][1][1](q);
            const T n1 = n_up_1(q);
            const T n2 = n_up_2(q);
            const T n1n1 = n1 * n1;
            const T n2n2 = n2 * n2;
            const T two_n1n2 = T(2) * n1 * n2;
            for (Index i = 0; i < n; ++i) {
                // Covariant Hessian rows: N_{|αβ} = N_{,αβ} − Γ^γ_{αβ} N_{,γ}.
                const T H11 = N_uu(q, i) - G1_11 * N_u(q, i) - G2_11 * N_v(q, i);
                const T H12 = N_uv(q, i) - G1_12 * N_u(q, i) - G2_12 * N_v(q, i);
                const T H22 = N_vv(q, i) - G1_22 * N_u(q, i) - G2_22 * N_v(q, i);
                C(q, i * ndof + slot) = n1n1 * H11 + two_n1n2 * H12 + n2n2 * H22;
            }
        }
        return C;
    }

private:
    std::size_t dof_index_;
};

/**
 * @brief Transverse displacement w.
 */
template <std::floating_point T>
class TransverseDisplacement : public BoundaryField<T>
{
public:

    Matrix<T> evaluate(const Element<T, 2>& element,
                       const PatchBoundary<T, 2>& boundary,
                       Index /*boundary_span*/,
                       const BasisDerivs<T, 1>& /*boundary_basis*/,
                       const IntrinsicGeometry<T, 1>& /*boundary_local*/,
                       Index /*parent_flat_span*/,
                       const BasisDerivs<T, 2>& parent_basis,
                       const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        return element.displacement_shape_matrix(*boundary.parent(),
                                                 parent_basis, parent_ig);
    }
};

/**
 * @brief Rotation projected onto the outward boundary normal: θ_n = n · θ.
 *
 * Computed intrinsically as `n^α θ_α` where θ_α are the element's covariant
 * rotation components (returned by `rotation_shape_matrix` with row 2q = θ_1,
 * row 2q+1 = θ_2) and n^α are the contravariant components of the boundary
 * outward in-surface normal.
 */
template <std::floating_point T>
class NormalRotation : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& boundary_local,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        const Matrix<T> Nrot = element.rotation_shape_matrix(
            *boundary.parent(), parent_basis, parent_ig);
        const ColMatrix<T, 3> n =
            boundary.eval_outward_normal(boundary_local, parent_ig);
        const auto [n_up_1, n_up_2] =
            detail::contravariant_components(n, parent_ig);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nrot.cols());
        for (Index q = 0; q < Q; ++q) {
            C.row(q) = n_up_1(q) * Nrot.row(2 * q)
                     + n_up_2(q) * Nrot.row(2 * q + 1);
        }
        return C;
    }
};

/**
 * @brief Rotation projected onto the boundary tangent: θ_s = s · θ, with
 *        s = a_3 × n the in-surface unit tangent 90° CCW from the outward
 *        normal.  Reduces to `(-n_y, n_x, 0)` on flat plates.
 *
 * Computed intrinsically as `s^α θ_α` (contravariant tangent × covariant
 * rotation).
 */
template <std::floating_point T>
class TangentialRotation : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& boundary_local,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        const Matrix<T> Nrot = element.rotation_shape_matrix(
            *boundary.parent(), parent_basis, parent_ig);
        const ColMatrix<T, 3> n =
            boundary.eval_outward_normal(boundary_local, parent_ig);
        const ExtrinsicGeometry<T, 2, 3> eg(parent_ig);
        const ColMatrix<T, 3>& a_3 = eg.n;
        const ColMatrix<T, 3> s = detail::surface_tangent(n, a_3);
        const auto [s_up_1, s_up_2] =
            detail::contravariant_components(s, parent_ig);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nrot.cols());
        for (Index q = 0; q < Q; ++q) {
            C.row(q) = s_up_1(q) * Nrot.row(2 * q)
                     + s_up_2(q) * Nrot.row(2 * q + 1);
        }
        return C;
    }
};

/**
 * @brief Normal transverse shear force: Q_n = n · Q.
 *
 * Reads the transverse-shear rows from `stress_matrix`. Plate-style elements
 * expose 5 strain rows per qp ordered (M^11, M^22, M^12, Q^1, Q^2); the
 * shear rows are 5q+3 (Q^1) and 5q+4 (Q^2). Then Q_n = n_α Q^α with n_α the
 * covariant components of the boundary outward in-surface normal.
 *
 * Not applicable to shear-rigid elements (Kirchhoff–Love), whose stress
 * vector has no shear rows.
 */
template <std::floating_point T>
class NormalTransverseShear : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& boundary_local,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        const Matrix<T> Nsigma = element.stress_matrix(
            *boundary.parent(), parent_basis, parent_ig);
        const ColMatrix<T, 3> n =
            boundary.eval_outward_normal(boundary_local, parent_ig);
        const auto [n_cov_1, n_cov_2] =
            detail::covariant_components(n, parent_ig);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nsigma.cols());
        for (Index q = 0; q < Q; ++q) {
            C.row(q) = n_cov_1(q) * Nsigma.row(5 * q + 3)
                     + n_cov_2(q) * Nsigma.row(5 * q + 4);
        }
        return C;
    }
};

/**
 * @brief Normal bending moment: M_nn = n_α n_β M^{αβ}.
 *
 * Reads the bending rows from `stress_matrix` in symmetric-Voigt contravariant
 * order (M^11, M^22, M^12), at rows 5q, 5q+1, 5q+2. The factor of 2 on the
 * off-diagonal accounts for the symmetric contribution M^12 = M^21. Reduces
 * to `m_x n_x² + m_y n_y² + 2 m_xy n_x n_y` on flat plates.
 */
template <std::floating_point T>
class NormalBendingMoment : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& boundary_local,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        const Matrix<T> Nsigma = element.stress_matrix(
            *boundary.parent(), parent_basis, parent_ig);
        const ColMatrix<T, 3> n =
            boundary.eval_outward_normal(boundary_local, parent_ig);
        const auto [n_cov_1, n_cov_2] =
            detail::covariant_components(n, parent_ig);
        const Index Q = static_cast<Index>(n.rows());
        const Index n_strain = Nsigma.rows() / Q;
        Matrix<T> C(Q, Nsigma.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n1 = n_cov_1(q);
            const T n2 = n_cov_2(q);
            C.row(q) = (n1 * n1)      * Nsigma.row(n_strain * q)
                     + (n2 * n2)      * Nsigma.row(n_strain * q + 1)
                     + T(2) * n1 * n2 * Nsigma.row(n_strain * q + 2);
        }
        return C;
    }
};

/**
 * @brief Twisting (cross) moment: M_{ns} = n_α s_β M^{αβ}, with s = a_3 × n.
 *
 * Work-conjugate to the tangential rotation θ_s on a Reissner–Mindlin
 * boundary; used as the traction in Nitsche enforcement of θ_s = θ̄_s.
 * Reads the bending rows from `stress_matrix` in symmetric-Voigt
 * contravariant order (M^11, M^22, M^12) at rows 5q, 5q+1, 5q+2 (or 3q for
 * pure-bending KL plates) and uses the symmetric form
 *   M_{ns} = n_1 s_1 M^11 + n_2 s_2 M^22 + (n_1 s_2 + n_2 s_1) M^12.
 */
template <std::floating_point T>
class TwistingMoment : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index /*boundary_span*/,
        const BasisDerivs<T, 1>& /*boundary_basis*/,
        const IntrinsicGeometry<T, 1>& boundary_local,
        Index /*parent_flat_span*/,
        const BasisDerivs<T, 2>& parent_basis,
        const IntrinsicGeometry<T, 2>& parent_ig) const override
    {
        const Matrix<T> Nsigma = element.stress_matrix(
            *boundary.parent(), parent_basis, parent_ig);
        const ColMatrix<T, 3> n =
            boundary.eval_outward_normal(boundary_local, parent_ig);
        const ExtrinsicGeometry<T, 2, 3> eg(parent_ig);
        const ColMatrix<T, 3>& a_3 = eg.n;
        const ColMatrix<T, 3> s = detail::surface_tangent(n, a_3);
        const auto [n_cov_1, n_cov_2] =
            detail::covariant_components(n, parent_ig);
        const auto [s_cov_1, s_cov_2] =
            detail::covariant_components(s, parent_ig);
        const Index Q = static_cast<Index>(n.rows());
        const Index n_strain = Nsigma.rows() / Q;
        Matrix<T> C(Q, Nsigma.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n1 = n_cov_1(q), n2 = n_cov_2(q);
            const T s1 = s_cov_1(q), s2 = s_cov_2(q);
            C.row(q) = (n1 * s1)            * Nsigma.row(n_strain * q)
                     + (n2 * s2)            * Nsigma.row(n_strain * q + 1)
                     + (n1 * s2 + n2 * s1)  * Nsigma.row(n_strain * q + 2);
        }
        return C;
    }
};

} // namespace pyck

#endif // PYCK_BOUNDARY_FIELD_HPP
