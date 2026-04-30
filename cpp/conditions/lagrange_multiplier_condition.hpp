#ifndef PYCK_LAGRANGE_MULTIPLIER_CONDITION_HPP
#define PYCK_LAGRANGE_MULTIPLIER_CONDITION_HPP

#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "bspline.hpp"
#include "condition.hpp"
#include "boundary_patch.hpp"
#include "dof_mapper.hpp"
#include "element.hpp"
#include "quadrature.hpp"
#include "../assembly/dof_layout.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Boundary Lagrange-multiplier condition for plate elements.
 *
 * Appends multiplier DOFs defined on the boundary trace basis and enforces
 * one or more constraints weakly but exactly in the augmented system:
 *
 *   [ K  C^T ] [u]   [f]
 *   [ C   0  ] [λ] = [g]
 *
 * where each active block in C corresponds to one boundary field:
 * transverse displacement w, normal rotation θ_n, or tangential rotation θ_s.
 * The multiplier field is discretized with the 1D boundary basis associated
 * with the supplied BoundaryPatch.
 *
 * Each enforced field declares its own DOF block on the global `DofLayout`
 * with a `DofType::LagrangeMultiplier` tag. (All fields use the same tag;
 * field identification relies on block ordering or external context.)
 *
 * @tparam T Scalar floating-point type (double or float).
 */
template <std::floating_point T>
class LagrangeMultiplierCondition : public Condition<T>
{
public:

    /**
     * @brief Construct a Lagrange-multiplier boundary condition.
     *
     * @param boundary        1D boundary patch extracted from a 2D surface.
     * @param element         2D element formulation providing trace operators.
     * @param quadrature      1D quadrature rule for the boundary integral.
     * @param enforce_w       If true, enforce the displacement trace w = w_bar.
     * @param w_bar           Prescribed displacement value.
     * @param enforce_phi_n   If true, enforce the normal rotation θ_n = phi_n_bar.
     * @param phi_n_bar       Prescribed normal rotation value.
     * @param enforce_phi_s   If true, enforce the tangential rotation θ_s = phi_s_bar.
     * @param phi_s_bar       Prescribed tangential rotation value.
     */
    LagrangeMultiplierCondition(const BoundaryPatch<T, 2>& boundary,
                                const Element<T, 2>& element,
                                const QuadratureRule<T, 1>& quadrature,
                                bool enforce_w = true, T w_bar = T(0),
                                bool enforce_phi_n = false, T phi_n_bar = T(0),
                                bool enforce_phi_s = false, T phi_s_bar = T(0));

    /**
     * @brief Report the number of Lagrange multiplier DOFs required.
     */
    std::size_t num_dofs() const override;

    /**
     * @brief Allocate multiplier DOF blocks in the layout.
     *
     * Called by the assembler during the DOF allocation phase. Stores the
     * allocated block IDs for use in apply().
     *
     * @param layout       Layout to allocate blocks against.
     * @param primal_block Handle to the primal block.
     */
    void allocate_dofs(DofLayout& layout, DofLayout::BlockId primal_block);

    /**
     * @brief Scatter the coupling blocks and prescribed trace RHS.
     *
     * @param stiffness    Global augmented stiffness matrix (modified in-place).
     * @param load         Global augmented load vector (modified in-place).
     * @param layout       Equation-numbering authority for the assembly.
     * @param primal_block Handle to the primal DOF block.
     */
    void apply(Matrix<T>& stiffness,
               Vector<T>& load,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

private:

    enum class ComponentKind {
        displacement,
        normal_rotation,
        tangential_rotation,
    };

    struct Component {
        ComponentKind kind;
        DofType dof_type;
        T prescribed_value;
        Index dof_count;
        bool use_derivative_basis;
        DofLayout::BlockId block_id = 0;  // set by declare_dofs
    };

    /**
     * @brief Per-span local contribution.
     *
     * `C` and `G` are stacked component-wise: rows for component[0] first,
     * then rows for component[1], etc. `elem_node_cps` is the list of
     * parent-patch CP indices for this span; the global primal DOF for
     * `(elem_node_cps[k], slot)` is resolved at apply time via
     * `layout.primal_dof(primal_block_, elem_node_cps[k], slot)`.
     * `basis_ids_per_component[c]` lists the block-local lambda IDs for
     * component `c` on this span; the global lambda DOF is
     * `layout.aux_dof(components_[c].block_id, basis_ids_per_component[c][i])`.
     */
    struct LocalContribution {
        Matrix<T> C;
        Vector<T> G;
        std::vector<Index> elem_node_cps;
        std::vector<std::vector<Index>> basis_ids_per_component;
    };

    std::vector<Component> components_;
    std::vector<LocalContribution> contributions_;
    Index boundary_basis_count_ = 0;
    Index derivative_basis_count_ = 0;
    Index node_dofs_ = 0;
    Ptr<const Basis<T>> derivative_basis_;
    std::optional<DofMapper<1>> derivative_dof_mapper_;
};

} // namespace pyck

#endif // PYCK_LAGRANGE_MULTIPLIER_CONDITION_HPP
