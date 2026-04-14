#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <array>
#include <vector>
#include <Eigen/Dense>

#include "patch.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Abstract base class for elements.
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
template <std::floating_point T>
class Element<T, 1>
{
public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const = 0;

    /// @brief Transverse-displacement shape matrix N_w (Q × K).
    ///
    /// Returns the fully-assembled shape matrix such that `w(q) = N_w(q,:) · û`
    /// where `û` is the full element DOF vector (length K = n · num_node_dofs).
    /// Columns corresponding to non-w slots are zero.  For formulations where
    /// w is not a primary DOF (e.g. bending-potential Timoshenko) this
    /// contains the effective shape function that recovers w from the
    /// formulation's DOFs.
    virtual Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Rotation shape matrix N_θ (Q × K).
    ///
    /// Returns the fully-assembled shape matrix such that `θ(q) = N_θ(q,:) · û`.
    /// For formulations with θ as an explicit DOF (e.g. Timoshenko 2p) the
    /// nonzero columns are at the θ slot; for formulations where θ is derived
    /// from w (Euler–Bernoulli, Timoshenko 1p) this returns −∂N/∂x acting on
    /// the w-field.
    virtual Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Number of DOFs per node
    virtual std::size_t num_node_dofs() const = 0;

    /// @brief Minimum required derivative order for shape function evaluation
    virtual std::size_t min_order() const { return 0; }

    /// @brief DOF slot (in [0, num_node_dofs)) at which the w-field lives.
    virtual std::size_t displacement_dof_index() const { return 0; }

    /// @brief DOF slot at which the θ-field lives.
    virtual std::size_t rotation_dof_index() const { return 0; }

protected:
    /// @brief Derivative mapping
    enum idx { val = 0, d1 = 1, d11 = 2, d111 = 3 };
};

/// @brief Specialized base class for 2D elements.
template <std::floating_point T>
class Element<T, 2>
{
public:
    virtual ~Element() = default;

    virtual void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                         const Vector<T>& jacobian,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const = 0;

    /// @brief Transverse-displacement shape matrix N_w (Q × K).
    ///
    /// Returns the fully-assembled shape matrix such that `w(q) = N_w(q,:) · û`
    /// where `û` is the full element DOF vector (length K = n · num_node_dofs).
    /// Columns corresponding to non-w slots are zero.  For formulations where
    /// w is not a primary DOF (e.g. single-variable Reissner–Mindlin with
    /// bending potential) this contains the effective shape function that
    /// recovers w from the formulation's DOFs.
    virtual Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    /// @brief Rotation shape matrix N_φ (2Q × K), row-stacked per quadrature
    /// point: row `2q` is the x-component (θ_x), row `2q+1` is the y-component
    /// (θ_y).  Returns the fully-assembled shape matrix such that
    /// `(θ_x, θ_y)(q) = N_φ(2q:2q+2, :) · û`.  The caller forms any directional
    /// projection, e.g. θ_n = n_x·row(2q) + n_y·row(2q+1).  For formulations
    /// where θ is derived from w (Kirchhoff–Love, single-variable
    /// Reissner–Mindlin), the nonzero columns are at the w-slot with values
    /// (−∂N/∂x, −∂N/∂y).
    virtual Matrix<T>
    rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const = 0;

    virtual std::size_t num_node_dofs() const = 0;

    virtual std::size_t min_order() const { return 0; }

    /// @brief DOF slot (in [0, num_node_dofs)) at which the w-field lives.
    ///
    /// For all plate formulations in pyck, w lives on DOF 0.
    virtual std::size_t displacement_dof_index() const { return 0; }

    /// @brief DOF slots (in [0, num_node_dofs)) at which (θ_x, θ_y) live.
    ///
    /// * RM-3p: (1, 2) — independent rotation DOFs.
    /// * KL-1p / RM-1p: (0, 0) — θ = ±∇w or ±∇w_b, both components act on
    ///   the same (sole) DOF.
    virtual std::array<std::size_t, 2> rotation_dof_indices() const = 0;

protected:
    /// @brief Derivative mapping
    enum idx { val = 0, d1 = 1, d2 = 2, d11 = 3, d12 = 4, d22 = 5, d111 = 6, d112 = 7, d122 = 8, d222 = 9 };
};

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
