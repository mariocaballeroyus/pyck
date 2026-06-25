#ifndef PYCK_PENALTY_COUPLING_CONDITION_HPP
#define PYCK_PENALTY_COUPLING_CONDITION_HPP

#include <Eigen/Dense>
#include <vector>

#include "boundary_value.hpp"
#include "condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Penalty coupling between two patches across a shared interface.
 *
 * @details Weakly enforces continuity of a field across the interface curve Γ
 *          shared by two patch boundaries, via the penalty form
 *          @f$ \alpha \int_\Gamma (w_A - w_B)(\delta w_A - \delta w_B)\,d\Gamma @f$,
 *          which assembles into the four blocks K_AA, K_AB, K_BA, K_BB. Side A
 *          drives the interface quadrature; the partner side B is evaluated at the
 *          coincident points via `BoundaryElementValues::reinit_on_boundary_pts`.
 *
 *          Non-conforming interfaces are supported: the two sides may have
 *          different knot vectors, element counts and degree. The constructor
 *          builds the common refinement of both sides' breakpoints into a list of
 *          integration segments, each lying within a single span on each side, so
 *          the product N_A·N_B integrates exactly. The A↔B point map is the affine
 *          arc-fraction map, which is exact for straight (and, more generally,
 *          affinely-parametrised) seams — the case built by Greville-placed control
 *          nets such as `SurfacePatch::rectangle`. A curved or
 *          mismatched-parametrisation seam (where the affine map is invalid) is
 *          detected and rejected pending a point-inversion driver. A conforming
 *          interface is just the degenerate case where the breakpoints coincide.
 *
 *          Couples the three displacement components (`U_X/U_Y/U_Z`, global axes,
 *          tied as a difference) and the normal/bending rotation (`ROT_N`). The
 *          frame-projected rotation is tied as a *sum*: the two sides' outward
 *          normals oppose at the seam (@f$ n_A = -n_B @f$), so the bending-rotation
 *          jump is @f$ \theta_A\!\cdot n_A + \theta_B\!\cdot n_B @f$. Tying `ROT_N`
 *          restores G1/slope continuity, so a bending plate transfers moment across
 *          the seam instead of hinging.
 *
 *          The hierarchic shear potential `PSI` of the four-parameter shells is also
 *          supported, tied as a *difference* (@f$ \psi_A - \psi_B @f$) like a global
 *          scalar — valid when both patches share the same surface-normal orientation
 *          (the ε-tensor in ψ's curl is built from that normal).
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Parent patch parametric dimension.
 */
template <std::floating_point T, std::size_t d>
requires (d > 1)
class PenaltyCouplingCondition : public Condition<T, d>
{
public:

    // === Constructors ===============================================================

    /**
     * @brief Build the coupling: validate the shared geometry and segment it.
     *
     * @param boundary_a One side of the shared interface.
     * @param boundary_b The other side (same physical curve; knots/degree may differ).
     * @param element    Shared element formulation supplying the field traces.
     * @param quadrature Boundary (d-1) quadrature, applied per integration segment.
     */
    PenaltyCouplingCondition(const PatchBoundary<T, d>& boundary_a,
                             const PatchBoundary<T, d>& boundary_b,
                             const Element<T, d>& element,
                             const QuadratureRule<T, d - 1>& quadrature);

    // === Utility ====================================================================

    /**
     * @brief Register a field whose continuity is penalised across the interface.
     *
     * @param field   Boundary value to tie: a displacement component
     *                (`U_X/U_Y/U_Z`), the normal/bending rotation (`ROT_N`) or the
     *                hierarchic shear potential (`PSI`) of the four-parameter shells.
     * @param penalty Penalty factor α.
     */
    PenaltyCouplingCondition& add(BoundaryValue<T> field, T penalty);

    /**
     * @brief Convenience: tie all three displacement components with one penalty.
     *
     * @param penalty Penalty factor α for `U_X`, `U_Y` and `U_Z`.
     */
    PenaltyCouplingCondition& couple_displacement(T penalty);

    /**
     * @brief Convenience: tie displacement and the normal (bending) rotation.
     *
     *        Adds `U_X/U_Y/U_Z` at @p penalty_disp and `ROT_N` at @p penalty_rot.
     *        This is the full G1 coupling for a shell/plate — displacement keeps the
     *        surface connected, `ROT_N` keeps it kink-free so bending moment crosses
     *        the seam. The two penalties scale differently (translation vs rotation)
     *        and are passed separately.
     *
     * @param penalty_disp Penalty factor for the displacement components.
     * @param penalty_rot  Penalty factor for the normal/bending rotation.
     */
    PenaltyCouplingCondition& couple_kinematics(T penalty_disp, T penalty_rot);

    /**
     * @brief Assemble the four coupling blocks into the global system.
     *
     * @param assembler  Sparse-assembly sink for stiffness and load (mutable).
     * @param layout     DOF layout (read-only).
     * @param blocks     Patch to primal-block resolver (resolves both patches).
     */
    void apply(SystemAssembler<T>& assembler,
               const DofLayout& layout,
               const PatchBlocks<T, d>& blocks) const override;

    // === Properties =================================================================

    /// @brief Routing patch: side A. The driver visits the condition once, under
    ///        A's primal block; `apply` resolves B's block via the resolver.
    const Patch<T, d>& patch() const override { return *boundary_a_.parent(); }

    /// @brief Patch owning side A of the interface.
    const Patch<T, d>& patch_a() const { return *boundary_a_.parent(); }

    /// @brief Patch owning side B of the interface.
    const Patch<T, d>& patch_b() const { return *boundary_b_.parent(); }

private:

    /// @brief A single penalised field continuity registered on the interface.
    struct Term {
        BoundaryValue<T> field;   ///< Boundary value whose jump is penalised.
        T penalty;                ///< Penalty weight α.
    };

    /// @brief One integration segment of the common refinement: a stretch of Γ
    ///        lying within a single span on each side, with the side-A and
    ///        (affinely-mapped) side-B quadrature points and the arc-length weights.
    struct Segment {
        Index a_span_live;        ///< Live boundary-span index on side A.
        Index b_span_live;        ///< Live boundary-span index on side B.
        ColMatrix<T, d - 1> a_pts;///< Side-A boundary parameters (Q × (d-1)).
        ColMatrix<T, d - 1> b_pts;///< Coincident side-B boundary parameters.
        Vector<T> weights;        ///< Quadrature weights over the segment in A-param.
    };

    /// @brief Affine arc-fraction map between the two boundary parametrisations.
    ColMatrix<T, d - 1> map_a_to_b(const ColMatrix<T, d - 1>& a_pts) const;
    T map_a_to_b(T a_param) const;
    T map_b_to_a(T b_param) const;

    /// @brief Live boundary-span index containing a boundary parameter.
    Index a_live_span(T a_param) const;
    Index b_live_span(T b_param) const;

    /// @brief Build the segment plan (common refinement + affine map) in the ctor.
    void build_segments();

    /// @brief The two interface sides.
    const PatchBoundary<T, d>& boundary_a_;
    const PatchBoundary<T, d>& boundary_b_;

    /// @brief Shared element formulation (supplies the field traces).
    const Element<T, d>& element_;

    /// @brief Boundary (d-1) quadrature rule, applied per integration segment.
    const QuadratureRule<T, d - 1>& quadrature_;

    /// @brief Registered penalised field continuities.
    std::vector<Term> terms_;

    /// @brief Integration segments (common refinement of both sides' breakpoints).
    std::vector<Segment> segments_;

    /// @brief Knot-span index → live boundary-span index, per side.
    std::vector<Index> a_span_inv_;
    std::vector<Index> b_span_inv_;

    /// @brief Whether side B is parametrised opposite to side A along Γ.
    bool reversed_ = false;

    /// @brief Coincidence tolerance (∝ edge length), for validation and the guard.
    T coincidence_tol_ = T(0);
};

} // namespace pyck

#endif // PYCK_PENALTY_COUPLING_CONDITION_HPP
