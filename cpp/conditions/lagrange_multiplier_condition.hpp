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

    std::size_t num_multipliers() const override { return num_multipliers_; }

    /**
     * @brief Scatter the coupling blocks and prescribed trace RHS.
     *
     * @param stiffness Global augmented stiffness matrix (modified in-place).
     * @param load      Global augmented load vector (modified in-place).
     */
    void apply(Matrix<T>& stiffness, Vector<T>& load) const override;

private:

    enum class ComponentKind {
        displacement,
        normal_rotation,
        tangential_rotation,
    };

    struct Component {
        ComponentKind kind;
        T prescribed_value;
        Index block_offset;
        Index dof_count;
        bool use_derivative_basis;
    };

    struct LocalContribution {
        Matrix<T> C;
        Vector<T> G;
        std::vector<Index> primal_dofs;
        std::vector<Index> lambda_dofs;
    };

    std::vector<Component> components_;
    std::vector<LocalContribution> contributions_;
    Index boundary_basis_count_ = 0;
    Index derivative_basis_count_ = 0;
    Index num_multipliers_ = 0;
    Ptr<const Basis<T>> derivative_basis_;
    std::optional<DofMapper<1>> derivative_dof_mapper_;
};

} // namespace pyck

#endif // PYCK_LAGRANGE_MULTIPLIER_CONDITION_HPP
