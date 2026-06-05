#include "load_boundary_condition.hpp"

#include "boundary_element_values.hpp"

#include <stdexcept>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
LoadBoundaryCondition<T, d>::LoadBoundaryCondition(const PatchBoundary<T, d>& boundary,
                                                   const Element<T, d>& element,
                                                   const QuadratureRule<T, d - 1>& quadrature)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature)
{}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
LoadBoundaryCondition<T, d>& LoadBoundaryCondition<T, d>::add(BoundaryValue<T> field, T value)
{
    terms_.push_back(Term{std::move(field), false, value, Vector<T>()});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
LoadBoundaryCondition<T, d>& LoadBoundaryCondition<T, d>::add(BoundaryValue<T> field,
                                                             const Vector<T>& values_at_qpts)
{
    const Index nq = num_active_qpts();

    if (values_at_qpts.size() != static_cast<Eigen::Index>(nq)) {
        throw std::runtime_error("LoadBoundaryCondition::add: "
                                 "values_at_qpts has size " +
                                 std::to_string(values_at_qpts.size()) +
                                 " but expected " + std::to_string(nq));
    }

    terms_.push_back(Term{std::move(field), true, T(0), values_at_qpts});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void LoadBoundaryCondition<T, d>::apply(SystemAssembler<T>& assembler,
                                        const DofLayout& layout,
                                        DofLayout::BlockId primal_block) const
{
    if (terms_.empty()) return;

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
    const Index n_elems  = bd_values.num_elements();
    const Index n_cps    = static_cast<Index>(bd_values.parent_vals_.act_pts_.rows());
    const Index n_primal = n_cps * element_.num_node_dofs();
    const Index n_gp     = static_cast<Index>(quadrature_.num_points());

    // --- Assemble Local Blocks ------------------------------------------------------
    // f_load = \int{ \bar{t} N_w^T d\Gamma }

    // Preallocate local load vector
    Vector<T> f_local(n_primal);

    // Loop over elements (spans)
    std::size_t qpt_offset = 0;
    for (Index s = 0; s < n_elems; ++s) {
        // Compute values at the boundary
        bd_values.reinit(s);

        // Reset the reused local load vector (all terms accumulate per span)
        f_local.setZero();

        // Loop over traction terms to apply
        for (const auto& term : terms_) {
            // Evaluate the field (w) at the boundary quadrature points
            // Boundary values (N_w) size = (n_gp,n_primal)
            Matrix<T> N_w = term.field.evaluate(element_, bd_values);

            // Loop over quadrature points
            for (Index q = 0; q < n_gp; ++q) {
                // Prescribed traction: constant or per-qp value
                const T t_bar = term.varying
                    ? term.values_at_qpts(static_cast<Eigen::Index>(qpt_offset + q))
                    : term.constant_value;
                // Get the boundary integration measure
                // ds = Jac * w_gp
                const T ds = bd_values.boundary_vals_.jac(q) *
                             bd_values.boundary_vals_.mapped_weights_(q);

                // Add contribution to the local load vector
                // f_load += \bar{t} * N_w * ds
                f_local.noalias() += (ds * t_bar) * N_w.row(q).transpose();
            }
        }
        qpt_offset += static_cast<std::size_t>(n_gp);

        // --- Scatter into Global System -------------------------------------------------
        // {f + f_load}

        // Scatter the parent primal DOFs for this span
        layout.scatter_primal(primal_block, bd_values.parent_vals_.elem_cps_,
                              bd_values.parent_vals_.elem_dofs_);

        // Scatter f_load into the global load vector
        for (Index i = 0; i < n_primal; ++i) {
            assembler.add_load(bd_values.parent_vals_.elem_dofs_[i], f_local(i));
        }
    }
}

// === Properties =====================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
Index LoadBoundaryCondition<T, d>::num_active_qpts() const
{
    return boundary_.tensor_product().num_elements()
         * static_cast<Index>(quadrature_.num_points());
}

template class LoadBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LoadBoundaryCondition<float, 2>;
#endif

} // namespace pyck
