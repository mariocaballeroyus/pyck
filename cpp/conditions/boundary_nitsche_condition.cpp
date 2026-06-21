#include "boundary_nitsche_condition.hpp"

#include "boundary_element_values.hpp"

#include <stdexcept>
#include <vector>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
NitscheBoundaryCondition<T, d>::NitscheBoundaryCondition(const PatchBoundary<T, d>& boundary,
                                                         const Element<T, d>& element,
                                                         const QuadratureRule<T, d - 1>& quadrature)
    : boundary_(boundary),
      element_(element),
      quadrature_(quadrature)
{}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
NitscheBoundaryCondition<T, d>& NitscheBoundaryCondition<T, d>::add(BoundaryValue<T> field,
                                                                    T penalty, T value)
{
    terms_.push_back({std::move(field), penalty, value});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
void NitscheBoundaryCondition<T, d>::apply(SystemAssembler<T>& assembler,
                                           const DofLayout& layout,
                                           const PatchBlocks<T, d>& blocks) const
{
    if (terms_.empty()) return;

    const DofLayout::BlockId primal_block = blocks.primal(*boundary_.parent());

    // Infer the required order and flags.
    Index    order = element_.basis_order();
    unsigned flags = element_.natural_flags();
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
    // K_nit = \int{ ( -N_w^T B_t - B_t^T N_w + \alpha N_w^T N_w ) d\Gamma }
    // f_nit = \int{ ( -B_t^T \bar{w} + \alpha N_w^T \bar{w} ) d\Gamma }
   
    // Local Nitsche stiffness and load, reused across spans
    Matrix<T> K_nit(n_primal, n_primal);
    Vector<T> f_nit(n_primal);

    // Loop over elements (spans)
    for (Index s = 0; s < n_elems; ++s) {
        // Compute values at the boundary
        bd_values.reinit(s);

        // Reset the reused local blocks
        K_nit.setZero();
        f_nit.setZero();

        // Global primal DOF indices for this span (needed for the auxiliary cross-terms).
        layout.scatter_primal(primal_block, bd_values.parent_vals_.elem_cps_,
                              bd_values.parent_vals_.elem_dofs_);
        const std::vector<Index>& gdofs = bd_values.parent_vals_.elem_dofs_;

        for (const auto& term : terms_) {
            // Kinematic trace N_w (n_gp × n_primal) and its conjugate flux. For a mixed
            // element the flux splits: B_t (n_gp × n_primal) carries the displacement part
            // (zero membrane + real shear), B_t_aux (n_gp × n_aux) the ε̃-sourced membrane.
            Matrix<T>          B_t_aux;
            std::vector<Index> aux_dofs;
            Matrix<T> N_w = term.field.evaluate(element_, bd_values);
            Matrix<T> B_t = term.field.evaluate_flux(element_, bd_values, B_t_aux, aux_dofs);

            const Index na = static_cast<Index>(aux_dofs.size());
            Matrix<T> K_cross = Matrix<T>::Zero(n_primal, na);   // K(u_test, ε̃)
            Vector<T> f_aux   = Vector<T>::Zero(na);

            // Loop over quadrature points
            for (Index q = 0; q < n_gp; ++q) {
                // Boundary integration measure ds = Jac * w_gp
                const T ds = bd_values.boundary_vals_.jac(q) *
                             bd_values.boundary_vals_.mapped_weights_(q);

                const auto nw = N_w.row(q);
                const auto bt = B_t.row(q);

                // K_nit += ( -N_w^T B_t - B_t^T N_w + \alpha N_w^T N_w ) ds
                K_nit.noalias() -= ds * nw.transpose() * bt;
                K_nit.noalias() -= ds * bt.transpose() * nw;
                K_nit.noalias() += ds * term.penalty * nw.transpose() * nw;

                // f_nit += ( -B_t^T \bar{w} + \alpha N_w^T \bar{w} ) ds
                f_nit.noalias() -= ds * term.value * bt.transpose();
                f_nit.noalias() += ds * term.penalty * term.value * nw.transpose();

                // Consistency coupling to the auxiliary (ε̃) flux columns, if any.
                if (na > 0) {
                    K_cross.noalias() -= ds * nw.transpose() * B_t_aux.row(q);
                    f_aux.noalias()   -= ds * term.value * B_t_aux.row(q).transpose();
                }
            }

            // Scatter the symmetric primal↔auxiliary cross-coupling K(u, ε̃) and the
            // matching auxiliary load. The aux-aux block is zero (the kinematic trace has
            // no auxiliary component); a standard element yields na == 0, so this is inert.
            for (Index a = 0; a < na; ++a) {
                const Index ga = aux_dofs[static_cast<std::size_t>(a)];
                for (Index i = 0; i < n_primal; ++i) {
                    assembler.add_stiffness(gdofs[i], ga, K_cross(i, a));
                    assembler.add_stiffness(ga, gdofs[i], K_cross(i, a));
                }
                assembler.add_load(ga, f_aux(a));
            }
        }

        // --- Scatter the primal Nitsche block into the global system ----------------
        for (Index i = 0; i < n_primal; ++i) {
            assembler.add_load(gdofs[i], f_nit(i));
            for (Index j = 0; j < n_primal; ++j)
                assembler.add_stiffness(gdofs[i], gdofs[j], K_nit(i, j));
        }
    }
}

template class NitscheBoundaryCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class NitscheBoundaryCondition<float, 2>;
#endif

} // namespace pyck
