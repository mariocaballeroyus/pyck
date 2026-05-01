#ifndef PYCK_BOUNDARY_FIELD_HPP
#define PYCK_BOUNDARY_FIELD_HPP

#include <concepts>
#include <vector>

#include "../elements/element.hpp"
#include "../geometry/patch_boundary.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Polymorphic boundary-field operator.
 *
 * A BoundaryField builds the Q×K constraint-row matrix that picks out a
 * scalar quantity (e.g. transverse displacement, normal rotation) from the
 * element's shape functions at each quadrature point on the boundary.
 *
 * Fields that need the outward normal derive it lazily from the boundary
 * patch via PatchBoundary::eval_outward_normal; fields that don't (e.g.
 * transverse displacement) ignore the geometric inputs entirely.
 */
template <std::floating_point T>
class BoundaryField
{
public:
    virtual ~BoundaryField() = default;

    virtual Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index boundary_span,
        const std::vector<Matrix<T>>& boundary_derivs,
        Index parent_flat_span,
        const std::vector<Matrix<T>>& parent_derivs) const = 0;
};

/**
 * @brief Transverse displacement w.
 */
template <std::floating_point T>
class TransverseDisplacement : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& /*boundary*/,
        Index /*boundary_span*/,
        const std::vector<Matrix<T>>& /*boundary_derivs*/,
        Index /*parent_flat_span*/,
        const std::vector<Matrix<T>>& parent_derivs) const override
    {
        return element.displacement_shape_matrix(parent_derivs);
    }
};

/**
 * @brief Rotation projected onto the outward boundary normal: θ_n = n·θ.
 */
template <std::floating_point T>
class NormalRotation : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index boundary_span,
        const std::vector<Matrix<T>>& boundary_derivs,
        Index parent_flat_span,
        const std::vector<Matrix<T>>& parent_derivs) const override
    {
        const Matrix<T> Nrot = element.rotation_shape_matrix(parent_derivs);
        const ColMatrix<T, 3> n = boundary.eval_outward_normal(
            boundary_span, boundary_derivs, parent_flat_span, parent_derivs);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nrot.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n_x = n(q, 0);
            const T n_y = n(q, 1);
            // Nrot stores [θ_x; θ_y] shape rows interleaved per quadrature point,
            // so row 2q = N_θx and row 2q+1 = N_θy. θ_n = n_x·θ_x + n_y·θ_y.
            C.row(q) = n_x * Nrot.row(2 * q) + n_y * Nrot.row(2 * q + 1);
        }
        return C;
    }
};

/**
 * @brief Rotation projected onto the boundary tangent: θ_s = s·θ, with
 *        s = (-n_y, n_x) the in-plane 90° rotation of the outward normal.
 */
template <std::floating_point T>
class TangentialRotation : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index boundary_span,
        const std::vector<Matrix<T>>& boundary_derivs,
        Index parent_flat_span,
        const std::vector<Matrix<T>>& parent_derivs) const override
    {
        const Matrix<T> Nrot = element.rotation_shape_matrix(parent_derivs);
        const ColMatrix<T, 3> n = boundary.eval_outward_normal(
            boundary_span, boundary_derivs, parent_flat_span, parent_derivs);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nrot.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n_x = n(q, 0);
            const T n_y = n(q, 1);
            // Tangent s = (-n_y, n_x) is the outward normal rotated 90° CCW.
            // θ_s = -n_y·θ_x + n_x·θ_y, with rows interleaved as in NormalRotation.
            C.row(q) = -n_y * Nrot.row(2 * q) + n_x * Nrot.row(2 * q + 1);
        }
        return C;
    }
};

/**
 * @brief Normal transverse shear force: q_n = n · q.
 *
 * Projects the transverse shear force vector onto the outward boundary normal.
 */
template <std::floating_point T>
class NormalTransverseShear : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index boundary_span,
        const std::vector<Matrix<T>>& boundary_derivs,
        Index parent_flat_span,
        const std::vector<Matrix<T>>& parent_derivs) const override
    {
        const Matrix<T> Nq = element.transverse_shear_matrix(parent_derivs);
        const ColMatrix<T, 3> n = boundary.eval_outward_normal(
            boundary_span, boundary_derivs, parent_flat_span, parent_derivs);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nq.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n_x = n(q, 0);
            const T n_y = n(q, 1);
            // Nq stores [q_x; q_y] shape rows interleaved. q_n = n·q = n_x·q_x + n_y·q_y.
            C.row(q) = n_x * Nq.row(2 * q) + n_y * Nq.row(2 * q + 1);
        }
        return C;
    }
};

/**
 * @brief Normal bending moment: m_n = n^T · M · n.
 *
 * Projects the moment tensor onto the outward boundary normal.
 * For a moment tensor M = [m_x, m_xy; m_xy, m_y] (with m_xy as twisting),
 * m_n = m_x·n_x² + 2·m_xy·n_x·n_y + m_y·n_y²
 */
template <std::floating_point T>
class NormalBendingMoment : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index boundary_span,
        const std::vector<Matrix<T>>& boundary_derivs,
        Index parent_flat_span,
        const std::vector<Matrix<T>>& parent_derivs) const override
    {
        const Matrix<T> Nm = element.moment_matrix(parent_derivs);
        const ColMatrix<T, 3> n = boundary.eval_outward_normal(
            boundary_span, boundary_derivs, parent_flat_span, parent_derivs);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nm.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n_x = n(q, 0);
            const T n_y = n(q, 1);
            const T n_x2 = n_x * n_x;
            const T n_y2 = n_y * n_y;
            // Nm rows per quadrature point follow Voigt order: [m_x, m_y, m_xy].
            // m_n = n^T M n = m_x·n_x² + m_y·n_y² + 2·m_xy·n_x·n_y.
            // Factor of 2 on m_xy accounts for both off-diagonal contributions of the symmetric tensor.
            C.row(q) = n_x2 * Nm.row(3 * q) + T(2) * n_x * n_y * Nm.row(3 * q + 2) + n_y2 * Nm.row(3 * q + 1);
        }
        return C;
    }
};

/**
 * @brief Twisting (cross) moment: m_{ns} = n^T · M · s, where
 *        s = (-n_y, n_x) is the boundary tangent (90° CCW of outward normal).
 *
 * For a moment tensor M = [m_x, m_xy; m_xy, m_y]:
 *   m_{ns} = n_x·n_y·(m_y - m_x) + (n_x² - n_y²)·m_xy.
 *
 * Work-conjugate to the tangential rotation φ_s on a Reissner–Mindlin
 * boundary; used as the traction in Nitsche enforcement of φ_s = φ̄_s.
 */
template <std::floating_point T>
class TwistingMoment : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(
        const Element<T, 2>& element,
        const PatchBoundary<T, 2>& boundary,
        Index boundary_span,
        const std::vector<Matrix<T>>& boundary_derivs,
        Index parent_flat_span,
        const std::vector<Matrix<T>>& parent_derivs) const override
    {
        const Matrix<T> Nm = element.moment_matrix(parent_derivs);
        const ColMatrix<T, 3> n = boundary.eval_outward_normal(
            boundary_span, boundary_derivs, parent_flat_span, parent_derivs);
        const Index Q = static_cast<Index>(n.rows());
        Matrix<T> C(Q, Nm.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n_x = n(q, 0);
            const T n_y = n(q, 1);
            const T nx_ny = n_x * n_y;
            const T diff_sq = n_x * n_x - n_y * n_y;
            // Nm rows: 3q = m_x, 3q+1 = m_y, 3q+2 = m_xy.
            C.row(q) = nx_ny * (Nm.row(3 * q + 1) - Nm.row(3 * q))
                     + diff_sq * Nm.row(3 * q + 2);
        }
        return C;
    }
};

} // namespace pyck

#endif // PYCK_BOUNDARY_FIELD_HPP
