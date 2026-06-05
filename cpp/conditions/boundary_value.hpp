#ifndef PYCK_BOUNDARY_VALUE_HPP
#define PYCK_BOUNDARY_VALUE_HPP

#include <concepts>

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
    ROT_X, ROT_Y, ROT_Z, ROT_N, ROT_S,
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
            case Field::ROT_X: element.rotation(parent, x, out); return out;
            case Field::ROT_Y: element.rotation(parent, y, out); return out;
            case Field::ROT_Z: element.rotation(parent, z, out); return out;
            case Field::ROT_N: element.rotation(parent, n, out); return out;
            case Field::ROT_S: element.rotation(parent, s, out); return out;
        }
        __builtin_unreachable();
    }

    /// @brief Extra basis derivative order beyond the element's own. Uniform (0)
    ///        for the kinematic fields; the element's `basis_order` already covers
    ///        its shape matrices.
    Index basis_order() const { return Index(0); }

    /// @brief Parent-side geometry the projection reads: the covariant basis and
    ///        metric (Deriv1) for the contravariant raise, and the surface normal
    ///        (Normal) for the boundary frame {n, s}.
    unsigned flags() const { return Flags::Deriv1 | Flags::Normal; }

    /// @brief Parent-side flags the element's shape-matrix call triggers.
    unsigned element_flags(const Element<T, 2>& e) const { return e.essential_flags(); }

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
