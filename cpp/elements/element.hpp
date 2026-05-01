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

/// @brief Abstract base class for elements (generic).
/// @tparam T Scalar type.
/// @tparam d Dimension.
template <std::floating_point T, std::size_t d>
class Element
{

public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const = 0;

    virtual Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual std::size_t num_node_dofs() const = 0;

    virtual std::size_t min_order() const = 0;

};

/// @brief Specialized base class for 1D elements.
///
/// Concrete elements provide the bending and shear strain-displacement blocks
/// (Bb, Bs) plus the bending and shear stiffnesses (Kb, Ks). The base class
/// then composes the local stiffness matrix and any shape/strain matrix
/// derived from these as Bb^T Kb Bb + Bs^T Ks Bs etc.
template <std::floating_point T>
class Element<T, 1>
{
public:
    virtual ~Element() = default;

    /// @brief Bending strain-displacement matrix Bb (Q x K), one row per quadrature point.
    virtual Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Shear strain-displacement matrix Bs (Q x K), one row per quadrature point.
    /// Formulations with no transverse shear (e.g. Euler-Bernoulli) must explicitly
    /// return a zero matrix.
    virtual Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Bending stiffness scalar Kb (e.g. EI).
    virtual T bending_stiffness() const = 0;

    /// @brief Shear stiffness scalar Ks (e.g. kGA). Defaults to 0 for shear-free formulations.
    virtual T shear_stiffness() const { return T(0); }

    /// @brief Concatenated strain-displacement matrix (vertically stacked Bb, Bs per quadrature point).
    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
    {
        Matrix<T> Bb = bending_strain_matrix(shape_derivs);
        Matrix<T> Bs = shear_strain_matrix(shape_derivs);
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

    /// @brief Default local stiffness assembly: Ke = sum_q dV[q] * (Kb Bb^T Bb + Ks Bs^T Bs).
    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const
    {
        const Matrix<T> Bb = bending_strain_matrix(shape_fns);
        const Matrix<T> Bs = shear_strain_matrix(shape_fns);
        const T Kb = bending_stiffness();
        const T Ks = shear_stiffness();

        const Index Q = q_weights.size();
        const Index K = Bb.cols();
        const bool has_shear = (Bs.rows() > 0);
        const Vector<T> dV = q_weights.cwiseProduct(jacobian);

        stiffness.setZero(K, K);
        for (Index q = 0; q < Q; ++q) {
            const auto bb = Bb.row(q);
            stiffness.noalias() += (dV[q] * Kb) * (bb.transpose() * bb);
            if (has_shear) {
                const auto bs = Bs.row(q);
                stiffness.noalias() += (dV[q] * Ks) * (bs.transpose() * bs);
            }
        }
    }

    /** @brief Transverse-displacement shape matrix N_w (Q x K). */
    virtual Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /** @brief Rotation shape matrix N_varphi (Q x K). */
    virtual Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Number of DOFs per node
    virtual std::size_t num_node_dofs() const = 0;

    /// @brief Minimum required derivative order for shape function evaluation
    virtual std::size_t min_order() const { return 0; }

    /// @brief DOF slot (in [0, num_node_dofs)) at which the w-field lives.
    virtual std::size_t displacement_dof_index() const { return 0; }

    /// @brief DOF slot at which the varphi-field lives.
    virtual std::size_t rotation_dof_index() const { return 0; }

protected:
    /// @brief Derivative mapping
    enum idx { val = 0, d1 = 1, d11 = 2, d111 = 3 };
};

/// @brief Specialized base class for 2D elements.
///
/// Concrete elements provide the bending and shear strain-displacement blocks
/// (Bb has 3 rows per quadrature point, Bs has 2) plus the bending and shear
/// constitutive matrices (Db is 3x3, Ds is 2x2). All other quantities — the
/// concatenated B, the local stiffness matrix, the moment recovery
/// Nm = Db Bb, and the shear-force recovery Nq = Ds Bs — are derived from
/// these by the base class.
template <std::floating_point T>
class Element<T, 2>
{
public:
    virtual ~Element() = default;

    /// @brief Bending strain-displacement matrix Bb (3Q x K), 3 rows per quadrature point.
    virtual Matrix<T> bending_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Shear strain-displacement matrix Bs (2Q x K), 2 rows per quadrature point.
    /// Formulations with no transverse shear strain (e.g. Kirchhoff-Love) must
    /// explicitly return a zero matrix.
    virtual Matrix<T> shear_strain_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Bending constitutive matrix Db (3 x 3).
    virtual Matrix<T> bending_constitutive_matrix() const = 0;

    /// @brief Shear constitutive matrix Ds (2 x 2). Defaults to zero for shear-free formulations.
    virtual Matrix<T> shear_constitutive_matrix() const
    {
        return Matrix<T>::Zero(2, 2);
    }

    /// @brief Concatenated strain-displacement matrix
    /// (per quadrature point: 3 bending rows then 2 shear rows, if any).
    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const
    {
        Matrix<T> Bb = bending_strain_matrix(shape_derivs);
        Matrix<T> Bs = shear_strain_matrix(shape_derivs);
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

    /// @brief Default local stiffness assembly: Ke = sum_q dV[q] * (Bb^T Db Bb + Bs^T Ds Bs).
    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const
    {
        const Matrix<T> Bb = bending_strain_matrix(shape_fns);
        const Matrix<T> Bs = shear_strain_matrix(shape_fns);
        const Matrix<T> Db = bending_constitutive_matrix();

        const Index Q = q_weights.size();
        const Index K = Bb.cols();
        const bool has_shear = (Bs.rows() > 0);
        const Matrix<T> Ds = has_shear ? shear_constitutive_matrix() : Matrix<T>(0, 0);
        const Vector<T> dV = q_weights.cwiseProduct(jacobian);

        stiffness.setZero(K, K);
        for (Index q = 0; q < Q; ++q) {
            const auto bb = Bb.middleRows(3 * q, 3);
            stiffness.noalias() += dV[q] * (bb.transpose() * Db * bb);
            if (has_shear) {
                const auto bs = Bs.middleRows(2 * q, 2);
                stiffness.noalias() += dV[q] * (bs.transpose() * Ds * bs);
            }
        }
    }

    /// @brief Moment-tensor shape matrix N_m (3Q x K), defined as Db Bb per quadrature point.
    virtual Matrix<T> moment_matrix(const std::vector<Matrix<T>>& shape_derivs) const
    {
        const Matrix<T> Bb = bending_strain_matrix(shape_derivs);
        const Matrix<T> Db = bending_constitutive_matrix();
        const Index Q = Bb.rows() / 3;
        const Index K = Bb.cols();
        Matrix<T> Nm(3 * Q, K);
        for (Index q = 0; q < Q; ++q) {
            Nm.middleRows(3 * q, 3).noalias() = Db * Bb.middleRows(3 * q, 3);
        }
        return Nm;
    }

    /// @brief Transverse-shear-force shape matrix N_q (2Q x K), defined as Ds Bs per quadrature point.
    /// Elements without an explicit shear strain (e.g. Kirchhoff-Love) must override this
    /// to provide an equilibrium-based recovery.
    virtual Matrix<T> transverse_shear_matrix(const std::vector<Matrix<T>>& shape_derivs) const
    {
        const Matrix<T> Bs = shear_strain_matrix(shape_derivs);
        if (Bs.rows() == 0) {
            throw std::runtime_error(
                "transverse_shear_matrix: element has no shear strain block; "
                "override transverse_shear_matrix() to provide an equilibrium recovery.");
        }
        const Matrix<T> Ds = shear_constitutive_matrix();
        const Index Q = Bs.rows() / 2;
        const Index K = Bs.cols();
        Matrix<T> Nq(2 * Q, K);
        for (Index q = 0; q < Q; ++q) {
            Nq.middleRows(2 * q, 2).noalias() = Ds * Bs.middleRows(2 * q, 2);
        }
        return Nq;
    }

    /** @brief Transverse-displacement shape matrix N_w (Q x K). */
    virtual Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /** @brief Rotation shape matrix N_varphi (2Q x K), row-stacked [varphi_x; varphi_y] per quadrature point. */
    virtual Matrix<T>
    rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual std::size_t num_node_dofs() const = 0;

    virtual std::size_t min_order() const { return 0; }

    /// @brief DOF slot at which the w-field lives.
    virtual std::size_t displacement_dof_index() const { return 0; }

    /// @brief DOF slots at which (varphi_x, varphi_y) live.
    virtual std::array<std::size_t, 2> rotation_dof_indices() const = 0;

protected:
    /// @brief Derivative mapping
    enum idx { val = 0, d1 = 1, d2 = 2, d11 = 3, d12 = 4, d22 = 5, d111 = 6, d112 = 7, d122 = 8, d222 = 9 };
};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
