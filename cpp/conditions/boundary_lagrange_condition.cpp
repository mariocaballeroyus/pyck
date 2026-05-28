#include "boundary_lagrange_condition.hpp"
#include "boundary_element_values.hpp"

#include <stdexcept>
#include <vector>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
LagrangeBoundaryCondition<T, d>::LagrangeBoundaryCondition(const PatchBoundary<T, d>& boundary,
                                                           const Element<T, d>& element,
                                                           const QuadratureRule<T, d - 1>& quadrature)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature),
      multiplier_dof_count_(static_cast<Index>(boundary.num_control_pts()))
{
    if (multiplier_dof_count_ == 0) {
        throw std::invalid_argument("LagrangeBoundaryCondition: "
                                    "boundary has no active basis functions.");
    }
}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
LagrangeBoundaryCondition<T, d>&
LagrangeBoundaryCondition<T, d>::add(Ptr<const BoundaryField<T>> field, T value)
{
    if (!field) {
        throw std::invalid_argument("LagrangeBoundaryCondition::add: "
                                    "field must not be null.");
    }
    terms_.push_back({std::move(field), value, 0});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void LagrangeBoundaryCondition<T, d>::allocate_dofs(DofLayout& layout)
{
    for (auto& term : terms_) {
        term.block_id = layout.allocate(DofType::LagrangeMultiplier, 
                                        multiplier_dof_count_, 1);
    }
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void
LagrangeBoundaryCondition<T, d>::apply(SystemAssembler<T>& assembler,
                                       const DofLayout& layout,
                                       DofLayout::BlockId primal_block) const
{
    if (terms_.empty()) return;

    // Infer the required order and flags
    Index    order = element_.basis_order();
    unsigned flags       = Flags::None;
    for (const auto& term : terms_) {
        order = std::max(order, term.field->basis_order());
        flags |= term.field->flags();
        flags |= term.field->element_flags(element_);
    }

    // Get the number of elements, primal DOFs, multipliers and boundary Gauss points
    BoundaryElementValues<T, d> bd_values(boundary_, order, flags, quadrature_);
    const Index n_elems  = bd_values.num_elements();
    const Index n_cps    = static_cast<Index>(bd_values.parent_vals_.act_pts_.rows());
    const Index n_primal = n_cps * element_.num_node_dofs();
    const Index n_lambda = static_cast<Index>(bd_values.boundary_vals_.act_pts_.rows());
    const Index n_gp     = static_cast<Index>(quadrature_.num_points());

    // --- Assemble Local Blocks ------------------------------------------------------
    // G = \int{ N_{\lambda}^T N_w d\Gamma } 
    // g = \int{ N_{\lambda}^T N_w \bar{w} d\Gamma }

    // Preallocate local coupling block and constraint vector
    Matrix<T> G_local(n_lambda, n_primal);
    Vector<T> g_local(n_lambda);

    // Loop over elements (spans)
    for (Index s = 0; s < n_elems; ++s) {
        // Compute values at the boundary
        bd_values.reinit(s);

        // Loop over Dirichlet values to enforce
        for (const auto& term : terms_) {
            // Evaluate the field (w) at the boundary quadrature points
            // Boundary values (N_w) size = (n_gp,n_primal)
            Matrix<T> N_w = term.field->evaluate(element_, bd_values);

            // Reset the reused local coupling block and constraint vector
            G_local.setZero();
            g_local.setZero();

            // Loop over quadrature points
            for (Index q = 0; q < n_gp; ++q) {
                // Get the boundary integration measure
                // ds = Jac * w_gp
                const T ds = bd_values.boundary_vals_.jac(q) * 
                             bd_values.boundary_vals_.mapped_weights_(q);
                // Get the multiplier shape
                auto N_lambda_i = bd_values.boundary_vals_.results_[0].col(q);

                // Add contributions to the local coupling block and constraint vector
                // G += N_λ * N_w * ds
                G_local.noalias() += N_lambda_i * N_w.row(q) * ds;
                // g += N_λ * \hat{w} * ds
                g_local.noalias() += N_lambda_i * term.value * ds;
            }

            // --- Scatter into Global System -----------------------------------------
            // [ K   G^T ] {\hat{w}} = {f}
            // [ G   0   ] {\hat{λ}} = {g}

            // Scatter the parent primal DOFs for this span
            layout.scatter_primal(primal_block, bd_values.parent_vals_.elem_cps_,
                                  bd_values.parent_vals_.elem_dofs_);

            // Starting index of muliplier-type DOFs
            const Index multiplier_base = layout.block_base(term.block_id);

            // Loop over Lagrange conditions
            for (Index i = 0; i < n_lambda; ++i) {
                // Scatter g into the global load vector
                const Index lambda_i = multiplier_base + bd_values.boundary_vals_.elem_cps_[i];
                assembler.add_load(lambda_i, g_local(i));
                // Scatter G into the global stiffness matrix
                for (Index j = 0; j < n_primal; ++j) {
                    const Index primal_j = bd_values.parent_vals_.elem_dofs_[j];
                    const T gij = G_local(i, j);
                    assembler.add_stiffness(lambda_i, primal_j, gij);  // G block
                    assembler.add_stiffness(primal_j, lambda_i, gij);  // G^T block
                }
            }
        }
    }
}

// === Template Instantiations ========================================================

template class LagrangeBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LagrangeBoundaryCondition<float, 2>;
#endif

} // namespace pyck
