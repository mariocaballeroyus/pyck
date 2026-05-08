#ifndef PYCK_LOAD_BOUNDARY_CONDITION_HPP
#define PYCK_LOAD_BOUNDARY_CONDITION_HPP

#include <vector>
#include <Eigen/Dense>

#include "boundary_field.hpp"
#include "condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Natural (Neumann) boundary condition: integrates the variational
 *        external-work term
 *
 *            deltaW_ext|Gamma_N = integral_{Gamma_N}  t_bar . delta v  dGamma
 *
 *        into the global load vector. `v` is a primal kinematic variable
 *        (e.g. transverse displacement w, normal rotation theta_n,
 *        tangential rotation theta_s, KL normal slope dw/dn) and
 *        `t_bar` is its work-conjugate prescribed traction:
 *
 *            field = TransverseDisplacement   ->   t_bar = shear Q_bar
 *            field = NormalRotation (RM)      ->   t_bar = bending moment M_n_bar
 *            field = TangentialRotation (RM)  ->   t_bar = twisting moment M_ns_bar
 *            field = BasisNormalSlope(0) (KL) ->   t_bar = bending moment M_n_bar
 *
 *        Per-quadrature-point contribution:
 *
 *            F_local += dGamma(q) * t_bar(q) * N_v(q,:)^T
 *
 *        where `N_v` is the Q x K row matrix supplied by `field->evaluate(...)`.
 *
 *        The class only writes to F; K and the auxiliary-DOF allocation are
 *        not touched. `t_bar` may be a constant scalar or pre-evaluated at the
 *        active boundary quadrature points.
 *
 * Notes on Kirchhoff-Love elements:
 * - Prescribing a twisting moment M_ns_bar directly as a single edge integral
 *   is incomplete on KL boundaries — the user must combine it with the shear
 *   into the Kirchhoff effective shear Q_eff_bar = Q_bar + d(M_ns_bar)/ds and
 *   add corner forces externally. This class performs no runtime check;
 *   callers are responsible for the physics.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch / element parametric dimension. Must be > 1; for
 *           `d == 1` the boundary is a 0-D point set, which is best handled
 *           via a dedicated point-load primitive (out of scope for v1).
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class LoadBoundaryCondition : public Condition<T>
{
public:
    LoadBoundaryCondition(const PatchBoundary<T, d>& boundary,
                          const Element<T, d>& element,
                          const QuadratureRule<T, d - 1>& quadrature);

    /// @brief Constant prescribed traction conjugate to `field`.
    LoadBoundaryCondition& add(Ptr<const BoundaryField<T>> field, T value);

    /// @brief Pre-evaluated per-quadrature-point traction values. Size must
    ///        equal `num_active_qpts()` (zero-length spans are skipped).
    LoadBoundaryCondition& add(Ptr<const BoundaryField<T>> field,
                               const Vector<T>& values_at_qpts);

    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               const std::vector<DofLayout::BlockId>& primal_blocks) const override;

    /// @brief Number of active boundary quadrature points (sum over
    ///        non-zero-length spans of `quadrature.num_points()`).
    Index num_active_qpts() const;

private:
    struct Term {
        Ptr<const BoundaryField<T>> field;
        bool varying = false;
        T constant_value = T(0);
        Vector<T> values_at_qpts;   // size == num_active_qpts() when varying
    };

    const PatchBoundary<T, d>& boundary_;
    const Element<T, d>& element_;
    const QuadratureRule<T, d - 1>& quadrature_;

    std::vector<Term> terms_;
};

} // namespace pyck

#endif // PYCK_LOAD_BOUNDARY_CONDITION_HPP
