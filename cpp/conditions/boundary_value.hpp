#ifndef PYCK_BOUNDARY_VALUE_HPP
#define PYCK_BOUNDARY_VALUE_HPP

#include <concepts>
#include <stdexcept>
#include <vector>

#include "../elements/element.hpp"
#include "../elements/boundary_element_values.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Boundary-value selector: which displacement / rotation component is
 *        traced on the boundary.
 *
 * `U_*` trace the recovered displacement, `ROT_*` the recovered rotation.
 * `_X/_Y/_Z` are global Cartesian components; `_N/_S` are the outward in-surface
 * normal and the in-surface tangent of the boundary. Whether a value is
 * prescribed data or a constraint is decided by the condition it is added to
 * (load vs penalty / Lagrange), not by the selector — a force in x and a clamp
 * of u_x are the same trace, work-conjugate to each other.
 */
enum class Field
{
    U_X, U_Y, U_Z, U_N, U_S,
    VB_X, VB_Y, VB_Z,   ///< Cartesian bending-surface displacement v_b (primal slots 0..2),
                        ///< without the shear correction the recovered displacement adds.
    ROT_X, ROT_Y, ROT_Z, ROT_N, ROT_S,
    DIR_N,    ///< Director-based normal rotation δa_3·n (surface-normal slope; C1 trace).
    KAPPA_NN, ///< Normal curvature n^α n^β κ_αβ (second fundamental form; C2 trace).
    PSI,   ///< Hierarchic shear potential ψ of the four-parameter shells (DOF slot 3).
    PSI_N, ///< Boundary-normal slope ψ_,n of the hierarchic field (the C1 trace of ψ).
};

/**
 * @brief A scalar boundary value: a named handle to one of the element's
 *        boundary-shape methods.
 *
 * Holds only the `Field` selector; `evaluate` picks the projection direction
 * for that field — a global Cartesian axis ê_x/ê_y/ê_z (`_X/_Y/_Z`) or the cached
 * boundary frame n / s (`_N/_S`) — and calls the element's `displacement` /
 * `rotation` boundary-shape method with it, getting back the scalar trace
 * directly. All the per-formulation projection knowledge lives in the element —
 * this is just the named layer over it. With `-Wswitch`, a new `Field` is a
 * compile error in `evaluate` until it is handled.
 */
template <std::floating_point T>
class BoundaryValue
{
public:

    explicit BoundaryValue(Field field) : field_(field) {}

    /// @brief The selected boundary-value field.
    Field field() const { return field_; }

    /// @brief Field-trace matrix (Q × K) at the boundary quadrature points.
    Matrix<T> evaluate(const Element<T, 2>& element,
                       const BoundaryElementValues<T, 2>& bvals) const
    {
        const ElementValues<T, 2>& parent = bvals.parent_vals_;
        const Index Q = parent.num_points();
        const ColMatrix<T, 3>& n = bvals.outward_normal();
        const ColMatrix<T, 3>& s = bvals.surface_tangent();
        const ColMatrix<T, 3>  x = cartesian_axis(Q, 0);
        const ColMatrix<T, 3>  y = cartesian_axis(Q, 1);
        const ColMatrix<T, 3>  z = cartesian_axis(Q, 2);
        Matrix<T> out;
        
        switch (field_) {
            case Field::U_X:   element.displacement(parent, x, out); return out;
            case Field::U_Y:   element.displacement(parent, y, out); return out;
            case Field::U_Z:   element.displacement(parent, z, out); return out;
            case Field::U_N:   element.displacement(parent, n, out); return out;
            case Field::U_S:   element.displacement(parent, s, out); return out;
            case Field::VB_X:  element.bending_displacement(parent, x, out); return out;
            case Field::VB_Y:  element.bending_displacement(parent, y, out); return out;
            case Field::VB_Z:  element.bending_displacement(parent, z, out); return out;
            case Field::ROT_X: element.rotation(parent, x, out); return out;
            case Field::ROT_Y: element.rotation(parent, y, out); return out;
            case Field::ROT_Z: element.rotation(parent, z, out); return out;
            case Field::ROT_N: element.rotation(parent, n, out); return out;
            case Field::ROT_S: element.rotation(parent, s, out); return out;
            case Field::DIR_N:    element.director_variation(parent, n, out); return out;
            case Field::KAPPA_NN: element.normal_curvature(parent, n, out);   return out;
            case Field::PSI:   element.psi(parent, out); return out;
            case Field::PSI_N: element.psi_gradient(parent, n, out); return out;
        }
        __builtin_unreachable();
    }

    /// @brief Work-conjugate flux-trace matrix (Q × K) at the boundary quadrature
    ///        points: the boundary force traction for the displacement fields (U_*)
    ///        or the bending moment for the rotation fields (ROT_*), built from the
    ///        element's generalised stress and contracted with the boundary co-normal
    ///        ν = `outward_normal()`. This is the consistency flux a Nitsche condition
    ///        pairs with `evaluate`; the projection direction is chosen exactly as there.
    Matrix<T> evaluate_flux(const Element<T, 2>& element,
                            const BoundaryElementValues<T, 2>& bvals) const
    {
        Matrix<T> out_aux;
        std::vector<Index> aux_dofs;
        return evaluate_flux(element, bvals, out_aux, aux_dofs);
    }

    /// @brief Flux trace as above, but also surfacing the element's **auxiliary-block**
    ///        flux columns @p out_aux (Q × n_aux) and their global DOF indices @p aux_dofs
    ///        — the ε̃-sourced membrane traction of a mixed element, which a Nitsche
    ///        condition pairs with the kinematic trace as the consistency coupling
    ///        K(u_test, ε̃). The work-conjugate moment fields (ROT_*) carry no membrane,
    ///        so their auxiliary part is empty.
    Matrix<T> evaluate_flux(const Element<T, 2>& element,
                            const BoundaryElementValues<T, 2>& bvals,
                            Matrix<T>& out_aux, std::vector<Index>& aux_dofs) const
    {
        const ElementValues<T, 2>& parent = bvals.parent_vals_;
        const Index Q = parent.num_points();
        const ColMatrix<T, 3>& n = bvals.outward_normal();   // boundary co-normal ν
        const ColMatrix<T, 3>& s = bvals.surface_tangent();
        const ColMatrix<T, 3>  x = cartesian_axis(Q, 0);
        const ColMatrix<T, 3>  y = cartesian_axis(Q, 1);
        const ColMatrix<T, 3>  z = cartesian_axis(Q, 2);
        Matrix<T> out;

        switch (field_) {
            case Field::U_X:   element.traction(parent, n, x, out, out_aux, aux_dofs); return out;
            case Field::U_Y:   element.traction(parent, n, y, out, out_aux, aux_dofs); return out;
            case Field::U_Z:   element.traction(parent, n, z, out, out_aux, aux_dofs); return out;
            case Field::U_N:   element.traction(parent, n, n, out, out_aux, aux_dofs); return out;
            case Field::U_S:   element.traction(parent, n, s, out, out_aux, aux_dofs); return out;
            case Field::VB_X: case Field::VB_Y: case Field::VB_Z:
                throw std::logic_error(
                    "BoundaryValue::evaluate_flux: the bending-surface displacement v_b "
                    "(VB_X/Y/Z) has no work-conjugate boundary flux; it is a multipatch "
                    "coupling trace only. Use U_X/Y/Z for the work-conjugate total "
                    "displacement traction.");
            case Field::ROT_X: element.moment(parent, n, x, out); break;
            case Field::ROT_Y: element.moment(parent, n, y, out); break;
            case Field::ROT_Z: element.moment(parent, n, z, out); break;
            case Field::ROT_N: element.moment(parent, n, n, out); break;
            case Field::ROT_S: element.moment(parent, n, s, out); break;
            case Field::DIR_N:
            case Field::KAPPA_NN:
                throw std::logic_error(
                    "BoundaryValue::evaluate_flux: the director rotation (DIR_N) and "
                    "normal curvature (KAPPA_NN) are multipatch coupling traces only; "
                    "they have no work-conjugate boundary flux.");
            case Field::PSI:
            case Field::PSI_N:
                throw std::logic_error(
                    "BoundaryValue::evaluate_flux: the hierarchic field psi has no "
                    "work-conjugate boundary flux (Nitsche/natural traces are undefined "
                    "for it). PSI/PSI_N are supported only as kinematic coupling traces.");
        }
        out_aux.resize(Q, 0);   // moment (ROT_*) fields: no membrane, empty auxiliary part
        aux_dofs.clear();
        return out;
    }

    /// @brief Extra basis derivative order beyond the element's own. Uniform (0)
    ///        for the kinematic fields; the element's `basis_order` already covers
    ///        its shape matrices.
    Index basis_order() const { return Index(0); }

    /// @brief Parent-side geometry the projection reads: the covariant basis and
    ///        metric (Deriv1) for the contravariant raise, and the surface normal
    ///        (Normal) for the boundary frame {n, s}.
    unsigned flags() const { return Flags::Deriv1 | Flags::Normal; }

    /// @brief Parent-side flags the element's shape-matrix call triggers. The normal
    ///        curvature reads the bending strain (the same geometry as the stress path),
    ///        so it needs the element's `natural_flags`; the kinematic traces need only
    ///        the `essential_flags`.
    unsigned element_flags(const Element<T, 2>& e) const
    {
        return field_ == Field::KAPPA_NN ? e.natural_flags() : e.essential_flags();
    }

private:

    /// @brief Constant Cartesian axis ê_k broadcast to the Q quadrature points
    ///        (Q × 3), so a global component is just a projection direction.
    static ColMatrix<T, 3> cartesian_axis(Index Q, Index k)
    {
        ColMatrix<T, 3> dir = ColMatrix<T, 3>::Zero(Q, 3);
        dir.col(k).setOnes();
        return dir;
    }

    Field field_;
};

} // namespace pyck

#endif // PYCK_BOUNDARY_VALUE_HPP
