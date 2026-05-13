#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <array>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

#include "patch.hpp"
#include "../types.hpp"

namespace pyck
{

/** @brief Abstract base class for elements (generic).
 * @tparam T Scalar type.
 * @tparam d Dimension.
 */
template <std::floating_point T, std::size_t d>
class Element
{

public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const Patch<T, d>& patch,
                                         Index elem_idx,
                                         const ColMatrix<T, d>& mapped_pts,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const = 0;

    virtual std::size_t num_node_dofs() const = 0;

    virtual std::size_t min_order() const = 0;

};

/// @brief Specialized base class for 1D elements.
///
/// Concrete elements provide the B-matrices (`bending_strain_matrix` /
/// `shear_strain_matrix`), the per-qp constitutives (`bending_constitutive` /
/// `shear_constitutive`), and the physical-field shape matrices
/// (`displacement_shape_matrix` / `rotation_shape_matrix`). The base class
/// owns the canonical local-stiffness assembly  K = ∫ (Bb^T Db Bb + Bs^T Ds Bs) dV
/// using these primitives.
template <std::floating_point T>
class Element<T, 1>
{
public:
    virtual ~Element() = default;

    /// @brief Bending strain-displacement matrix Bb (Q x K), one row per quadrature point.
    virtual Matrix<T> bending_strain_matrix(const Patch<T, 1>& patch,
                                            const BasisDerivs<T, 1>& basis,
                                            const LocalFrame<T, 1>& local) const = 0;

    /// @brief Shear strain-displacement matrix Bs (Q x K), one row per quadrature point.
    /// Formulations with no transverse shear (e.g. Euler-Bernoulli) must explicitly
    /// return a zero matrix.
    virtual Matrix<T> shear_strain_matrix(const Patch<T, 1>& patch,
                                          const BasisDerivs<T, 1>& basis,
                                          const LocalFrame<T, 1>& local) const = 0;

    /// @brief Bending constitutive Db at quadrature point q (e.g. EI · (g^{11})²).
    /// Per-qp because the metric raisings depend on the local frame.
    virtual T bending_constitutive(const LocalFrame<T, 1>& local, Index q) const = 0;

    /// @brief Shear constitutive Ds at quadrature point q. Defaults to 0 for
    /// shear-free formulations.
    virtual T shear_constitutive(const LocalFrame<T, 1>& /*local*/, Index /*q*/) const
    { return T(0); }

    /// @brief Combined strain-displacement matrix B (n_strain·Q × K), with
    /// `n_strain` rows per quadrature point. Default: bending row (and shear
    /// row, if any) interleaved per qp. Shells with cross-strain couplings
    /// override this to provide a richer block.
    virtual Matrix<T> strain_displacement_matrix(const Patch<T, 1>& patch,
                                                 const BasisDerivs<T, 1>& basis,
                                                 const LocalFrame<T, 1>& local) const
    {
        Matrix<T> Bb = bending_strain_matrix(patch, basis, local);
        Matrix<T> Bs = shear_strain_matrix(patch, basis, local);
        if (Bs.rows() == 0) return Bb;

        const Index Q = Bb.rows();
        const Index K = Bb.cols();
        Matrix<T> B(2 * Q, K);
        for (Index q = 0; q < Q; ++q) {
            B.row(2 * q    ) = Bb.row(q);
            B.row(2 * q + 1) = Bs.row(q);
        }
        return B;
    }

    /// @brief Combined constitutive D at quadrature point q, matching the
    /// row layout of `strain_displacement_matrix`. Default: block-diagonal
    /// `[Db ; Ds]`. Shells with cross-strain coupling override this.
    virtual Matrix<T> constitutive_matrix(const LocalFrame<T, 1>& local, Index q) const
    {
        const T Db = bending_constitutive(local, q);
        const T Ds = shear_constitutive(local, q);
        if (Ds == T(0)) {
            Matrix<T> D(1, 1);
            D(0, 0) = Db;
            return D;
        }
        Matrix<T> D = Matrix<T>::Zero(2, 2);
        D(0, 0) = Db;
        D(1, 1) = Ds;
        return D;
    }

    /// @brief Canonical local stiffness assembly: K = ∫ B^T D B dV. Concrete
    /// elements should rarely override this; provide the B and D matrices
    /// (directly, or via the bending/shear building blocks).
    virtual void compute_local_stiffness(const Patch<T, 1>& patch,
                                         Index elem_idx,
                                         const ColMatrix<T, 1>& mapped_pts,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const
    {
        auto basis   = patch.eval_basis(mapped_pts, elem_idx, min_order());
        auto act_pts = patch.active_control_pts(elem_idx);
        auto local   = patch.eval_local_frame(basis, act_pts);
        const Matrix<T> B = strain_displacement_matrix(patch, basis, local);
        const Index Q = q_weights.size();
        const Index n_strain = B.rows() / Q;
        const Index K = B.cols();
        stiffness.setZero(K, K);
        for (Index q = 0; q < Q; ++q) {
            const T dV = q_weights(q) * local.jac(q);
            const Matrix<T> D = constitutive_matrix(local, q);
            const auto B_q = B.middleRows(n_strain * q, n_strain);
            stiffness.noalias() += dV * (B_q.transpose() * D * B_q);
        }
    }

    /** @brief Transverse-displacement shape matrix N_w (Q x K). */
    virtual Matrix<T> displacement_shape_matrix(const Patch<T, 1>& patch,
                                                const BasisDerivs<T, 1>& basis,
                                                const LocalFrame<T, 1>& local) const = 0;

    /** @brief Rotation shape matrix N_varphi (Q x K). */
    virtual Matrix<T> rotation_shape_matrix(const Patch<T, 1>& patch,
                                            const BasisDerivs<T, 1>& basis,
                                            const LocalFrame<T, 1>& local) const = 0;

    /// @brief Number of DOFs per node
    virtual std::size_t num_node_dofs() const = 0;

    /// @brief Minimum required derivative order for shape function evaluation
    virtual std::size_t min_order() const { return 0; }

    /// @brief DOF slot (in [0, num_node_dofs)) at which the w-field lives.
    virtual std::size_t displacement_dof_index() const { return 0; }

    /// @brief DOF slot at which the varphi-field lives.
    virtual std::size_t rotation_dof_index() const { return 0; }
};

/// @brief Specialized base class for 2D elements.
///
/// Concrete elements provide bending and shear B-blocks plus surface-tensor
/// constitutives. All auxiliary methods take `(patch, basis, local)` and
/// compute Christoffel symbols internally via `patch.eval_christoffel(local)`
/// when needed.
template <std::floating_point T>
class Element<T, 2>
{
public:
    virtual ~Element() = default;

    /// @brief Bending strain-displacement matrix Bb (3Q x K), 3 rows per quadrature point.
    virtual Matrix<T> bending_strain_matrix(const Patch<T, 2>& patch,
                                            const BasisDerivs<T, 2>& basis,
                                            const LocalFrame<T, 2>& local) const = 0;

    /// @brief Shear strain-displacement matrix Bs (2Q x K), 2 rows per quadrature point.
    /// Formulations with no transverse shear strain (e.g. Kirchhoff-Love) must
    /// explicitly return a zero matrix.
    virtual Matrix<T> shear_strain_matrix(const Patch<T, 2>& patch,
                                          const BasisDerivs<T, 2>& basis,
                                          const LocalFrame<T, 2>& local) const = 0;

    /// @brief Bending constitutive matrix Db (3 x 3) at quadrature point q.
    /// Per-qp because the surface-tensor form depends on the inverse metric.
    virtual Matrix<T> bending_constitutive_matrix(const LocalFrame<T, 2>& local,
                                                  Index q) const = 0;

    /// @brief Shear constitutive matrix Ds (2 x 2) at quadrature point q.
    /// Defaults to zero for shear-free formulations.
    virtual Matrix<T> shear_constitutive_matrix(const LocalFrame<T, 2>& /*local*/,
                                                Index /*q*/) const
    {
        return Matrix<T>::Zero(2, 2);
    }

    /// @brief Combined strain-displacement matrix B (n_strain·Q × K), with
    /// `n_strain` rows per quadrature point. Default: 3 bending rows then
    /// 2 shear rows (if any) per qp. Shells with cross-strain couplings
    /// (membrane–bending, etc.) override this to provide a richer block.
    virtual Matrix<T> strain_displacement_matrix(const Patch<T, 2>& patch,
                                                 const BasisDerivs<T, 2>& basis,
                                                 const LocalFrame<T, 2>& local) const
    {
        Matrix<T> Bb = bending_strain_matrix(patch, basis, local);
        Matrix<T> Bs = shear_strain_matrix(patch, basis, local);
        if (Bs.rows() == 0) return Bb;

        const Index Q = Bb.rows() / 3;
        const Index K = Bb.cols();
        Matrix<T> B(5 * Q, K);
        for (Index q = 0; q < Q; ++q) {
            B.middleRows(5 * q,     3) = Bb.middleRows(3 * q, 3);
            B.middleRows(5 * q + 3, 2) = Bs.middleRows(2 * q, 2);
        }
        return B;
    }

    /// @brief Combined constitutive D at quadrature point q, matching the
    /// row layout of `strain_displacement_matrix`. Default: block-diagonal
    /// `[Db (3×3); Ds (2×2)]`. Shells with cross-strain coupling override.
    virtual Matrix<T> constitutive_matrix(const LocalFrame<T, 2>& local, Index q) const
    {
        const Matrix<T> Db = bending_constitutive_matrix(local, q);
        const Matrix<T> Ds = shear_constitutive_matrix(local, q);
        if (Ds.isZero()) return Db;
        Matrix<T> D = Matrix<T>::Zero(5, 5);
        D.topLeftCorner(3, 3)     = Db;
        D.bottomRightCorner(2, 2) = Ds;
        return D;
    }

    /// @brief Canonical local stiffness assembly: K = ∫ B^T D B dV. Concrete
    /// elements should rarely override this; provide the B and D matrices
    /// (directly, or via the bending/shear building blocks).
    virtual void compute_local_stiffness(const Patch<T, 2>& patch,
                                         Index elem_idx,
                                         const ColMatrix<T, 2>& mapped_pts,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const
    {
        auto basis   = patch.eval_basis(mapped_pts, elem_idx, min_order());
        auto act_pts = patch.active_control_pts(elem_idx);
        auto local   = patch.eval_local_frame(basis, act_pts);
        const Matrix<T> B = strain_displacement_matrix(patch, basis, local);
        const Index Q = q_weights.size();
        const Index n_strain = B.rows() / Q;
        const Index K = B.cols();
        stiffness.setZero(K, K);
        for (Index q = 0; q < Q; ++q) {
            const T dV = q_weights(q) * local.jac(q);
            const Matrix<T> D = constitutive_matrix(local, q);
            const auto B_q = B.middleRows(n_strain * q, n_strain);
            stiffness.noalias() += dV * (B_q.transpose() * D * B_q);
        }
    }

    /// @brief Moment-tensor shape matrix N_m (3Q x K), defined as Db Bb per quadrature point.
    virtual Matrix<T> moment_matrix(const Patch<T, 2>& patch,
                                    const BasisDerivs<T, 2>& basis,
                                    const LocalFrame<T, 2>& local) const
    {
        const Matrix<T> Bb = bending_strain_matrix(patch, basis, local);
        const Index Q = Bb.rows() / 3;
        const Index K = Bb.cols();
        Matrix<T> Nm(3 * Q, K);
        for (Index q = 0; q < Q; ++q) {
            const Matrix<T> Db = bending_constitutive_matrix(local, q);
            Nm.middleRows(3 * q, 3).noalias() = Db * Bb.middleRows(3 * q, 3);
        }
        return Nm;
    }

    /// @brief Transverse-shear-force shape matrix N_q (2Q x K), defined as Ds Bs per quadrature point.
    /// Elements without an explicit shear strain (e.g. Kirchhoff-Love) must override this
    /// to provide an equilibrium-based recovery.
    virtual Matrix<T> transverse_shear_matrix(const Patch<T, 2>& patch,
                                              const BasisDerivs<T, 2>& basis,
                                              const LocalFrame<T, 2>& local) const
    {
        const Matrix<T> Bs = shear_strain_matrix(patch, basis, local);
        if (Bs.rows() == 0) {
            throw std::runtime_error(
                "transverse_shear_matrix: element has no shear strain block; "
                "override transverse_shear_matrix() to provide an equilibrium recovery.");
        }
        const Index Q = Bs.rows() / 2;
        const Index K = Bs.cols();
        Matrix<T> Nq(2 * Q, K);
        for (Index q = 0; q < Q; ++q) {
            const Matrix<T> Ds = shear_constitutive_matrix(local, q);
            Nq.middleRows(2 * q, 2).noalias() = Ds * Bs.middleRows(2 * q, 2);
        }
        return Nq;
    }

    /** @brief Transverse-displacement shape matrix N_w (Q x K). */
    virtual Matrix<T> displacement_shape_matrix(const Patch<T, 2>& patch,
                                                const BasisDerivs<T, 2>& basis,
                                                const LocalFrame<T, 2>& local) const = 0;

    /** @brief Rotation shape matrix N_varphi (2Q x K), row-stacked [varphi_x; varphi_y] per quadrature point. */
    virtual Matrix<T> rotation_shape_matrix(const Patch<T, 2>& patch,
                                            const BasisDerivs<T, 2>& basis,
                                            const LocalFrame<T, 2>& local) const = 0;

    virtual std::size_t num_node_dofs() const = 0;

    virtual std::size_t min_order() const { return 0; }

    /// @brief DOF slot at which the w-field lives.
    virtual std::size_t displacement_dof_index() const { return 0; }

    /// @brief DOF slots at which (varphi_x, varphi_y) live.
    virtual std::array<std::size_t, 2> rotation_dof_indices() const = 0;
};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
