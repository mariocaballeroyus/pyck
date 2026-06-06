#include "boundary_penalty_condition.hpp"

#include "boundary_element_values.hpp"

#include <stdexcept>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyBoundaryCondition<T, d>::PenaltyBoundaryCondition(const PatchBoundary<T, d>& boundary,
                                                         const Element<T, d>& element,
                                                         const QuadratureRule<T, d - 1>& quadrature)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature)
{}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyBoundaryCondition<T, d>& PenaltyBoundaryCondition<T, d>::add(BoundaryValue<T> field,
                                                                    T penalty, T value)
{
    terms_.push_back({std::move(field), penalty, value});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void PenaltyBoundaryCondition<T, d>::apply(SystemAssembler<T>& assembler,
                                           const DofLayout& layout,
                                           const PatchBlocks<T, d>& blocks) const
{
    if (terms_.empty()) return;

    const DofLayout::BlockId primal_block = blocks.primal(*boundary_.parent());

    // Infer the required order and flags
    Index    order = element_.basis_order();
    unsigned flags = Flags::None;
    for (const auto& term : terms_) {
        order = std::max(order, term.field.basis_order());
        flags |= term.field.flags();
        flags |= term.field.element_flags(element_);
    }

    // Get the number of elements, primal DOFs and boundary Gauss points
    BoundaryElementValues<T, d> bd_values(boundary_, order, flags, quadrature_);
    const Index n_elems = bd_values.num_elements();
    const Index n_cps    = static_cast<Index>(bd_values.parent_vals_.act_pts_.rows());
    const Index n_primal = n_cps * element_.num_node_dofs();
    const Index n_gp     = static_cast<Index>(quadrature_.num_points());

    // --- Assemble Local Blocks ------------------------------------------------------
    // K_pen = \int{ \alpha N_w^T N_w d\Gamma }
    // f_pen = \int{ \alpha \bar{w} N_w d\Gamma }

    // Local penalty stiffness and load, reused across spans
    Matrix<T> K_pen(n_primal, n_primal);
    Vector<T> f_pen(n_primal);

    // Loop over elements (spans)
    for (Index s = 0; s < n_elems; ++s) {
        // Compute values at the boundary
        bd_values.reinit(s);

        // Reset the reused local blocks
        K_pen.setZero();
        f_pen.setZero();

        for (const auto& term : terms_) {
            // Evaluate the field (w) at the boundary quadrature points
            Matrix<T> N_w = term.field.evaluate(element_, bd_values);

            // Loop over quadrature points
            for (Index q = 0; q < n_gp; ++q) {
                // Get the boundary integration measure
                // ds = Jac * w_gp
                const T ds = bd_values.boundary_vals_.jac(q) *
                             bd_values.boundary_vals_.mapped_weights_(q);

                // Add constributions to the local stiffness and load penalties
                // K_pen += \alpha N_w^T N_w ds 
                K_pen.noalias() += ds * term.penalty
                                   * N_w.row(q).transpose() * N_w.row(q);
                // f_pen += \alpha \bar{w} N_w^T ds
                f_pen.noalias() += ds * term.penalty * term.value
                                   * N_w.row(q).transpose();
            }
        }

        // --- Scatter into Global System ---------------------------------------------
        // [ K + K_pen ] = {f + f_pen}

        // Scatter the parent primal DOFs for this span
        layout.scatter_primal(primal_block, bd_values.parent_vals_.elem_cps_,
                              bd_values.parent_vals_.elem_dofs_);

        // Loop over primal DOFs
        for (Index i = 0; i < n_primal; ++i) {
            // Scatter f_pen into global load vector
            const Index gi = bd_values.parent_vals_.elem_dofs_[i];
            assembler.add_load(gi, f_pen(i));
            // Scatter K_pen into global stiffness matrix
            for (Index j = 0; j < n_primal; ++j) {
                assembler.add_stiffness(gi, bd_values.parent_vals_.elem_dofs_[j], K_pen(i, j));
            }
        }
    }
}

template class PenaltyBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PenaltyBoundaryCondition<float, 2>;
#endif

} // namespace pyck
