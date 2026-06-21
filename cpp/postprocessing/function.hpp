#ifndef PYCK_FUNCTION_HPP
#define PYCK_FUNCTION_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../elements/element.hpp"
#include "../elements/element_values.hpp"
#include "../geometry/patch.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Callable wrapper around a DOF vector, an element formulation,
 *        and a patch. Evaluates the chosen field at parametric points.
 *
 * `operator()(params)` returns a `(Q, k)` matrix, where Q is the number of
 * evaluation points and k is the number of components produced per point by
 * the element shape matrix associated with @ref FieldType (1 for plate
 * displacement, 2 for plate rotation/shear, 3 for plate bending strain,
 * etc.).
 *
 * Single-patch only. The DOF vector must hold exactly
 * `patch.num_control_pts() * element.num_node_dofs()` entries (the physical
 * DOF slice — no Lagrange multipliers).
 *
 * @tparam T Scalar.
 * @tparam d Parametric dimension (1 = curve, 2 = surface).
 */
template <std::floating_point T, std::size_t d>
class Function
{
public:
    Function(Vector<T> u, Ptr<Element<T, d>> element, Ptr<Patch<T, d>> patch, FieldType field)
        : u_(std::move(u)), 
          element_(std::move(element)), 
          patch_(std::move(patch)), 
          field_(field)
    {
        // The physical (per-node) block is the leading slice of the solution. A bare
        // physical vector (== n_cp·ndof) drives the standard fields; the FULL solution
        // (> n_cp·ndof, incl. a mixed formulation's auxiliary DOFs) is required for the
        // faithful, ε̃-sourced membrane fields (see the two-block gather below).
        const Index phys = static_cast<Index>(patch_->num_control_pts())
                         * static_cast<Index>(element_->num_node_dofs());

        if (u_.size() < phys) {
            throw std::runtime_error(
                "Function: DOF vector has " + std::to_string(u_.size()) +
                " entries but the physical block needs " + std::to_string(phys) +
                " (n_cp * ndof).");
        }
    }

    /**
     * @brief Evaluate the field at parametric points.
     *
     * @param params `(Q, d)` parametric coordinates.
     * @return `(Q, k)` matrix of field values.
     */
    Matrix<T> operator()(const ColMatrix<T, d>& params) const
    {
        const Index Q = static_cast<Index>(params.rows());
        if (Q == 0) {
            return Matrix<T>(0, 0);
        }

        const Index ndof = static_cast<Index>(element_->num_node_dofs());

        // Per-point dispatcher: triggers the element's shape matrix computation
        // and returns a reference to its workspace (k × n_active·ndof).
        auto eval = [field = field_](const Element<T, d>& e,
                                     const ElementValues<T, d>& ev)
                                     -> const Matrix<T>& {
            switch (field) {
                case FieldType::PRIMAL:       e.primal_shape_matrix(ev);       return e.N_primal_;
                case FieldType::DISPLACEMENT: e.displacement_shape_matrix(ev); return e.N_w_;
                case FieldType::ROTATION:     e.rotation_shape_matrix(ev);     return e.N_phi_;
                case FieldType::STRAIN:       e.strain_matrix(ev);             return e.B_voigt_;
                case FieldType::TRACTION:     e.traction_shape_matrix(ev);     return e.N_traction_;
                case FieldType::MOMENT:       e.moment_shape_matrix(ev);       return e.N_moment_;
            }
            throw std::runtime_error("Function: unknown FieldType.");
        };

        // Single-point workspace, refreshed per parametric query via reinit_on_pts.
        // Evaluating directly per point — never materialise the global N matrix.
        ElementValues<T, d> ev(*patch_, element_->basis_order(),
                               element_->flags(), std::size_t(1));

        Matrix<T> out;
        Index     k = 0;
        Vector<T> u_local;

        // Two-block-gather scratch (auxiliary contribution; empty for standard elements).
        Matrix<T>          N_aux;
        std::vector<Index> aux_dofs;

        for (Index q = 0; q < Q; ++q)
        {
            ColMatrix<T, d> pt = params.row(q);

            std::array<Index, d> span_idx;
            for (std::size_t dir = 0; dir < d; ++dir)
                span_idx[dir] = patch_->basis(dir).find_span(
                    static_cast<T>(pt(0, static_cast<Index>(dir))));

            ev.reinit_on_pts(span_idx, pt);

            const Matrix<T>& N_local = eval(*element_, ev);

            if (q == 0) {
                k = static_cast<Index>(N_local.rows());
                out.resize(Q, k);
                u_local.resize(N_local.cols());
            }

            // Gather the active DOF subvector u_local from the global u_.
            const auto&  cps      = ev.elem_cps_;
            const Index  n_active = static_cast<Index>(cps.size());
            for (Index i = 0; i < n_active; ++i)
                for (Index v = 0; v < ndof; ++v)
                    u_local(i * ndof + v) = u_(cps[i] * ndof + v);

            // Local mat-vec: (k × n_active·ndof) × (n_active·ndof) — tight inner loop.
            out.row(q).noalias() = (N_local * u_local).transpose();

            // Two-block gather: an element (e.g. a mixed formulation) may add an
            // auxiliary contribution sourced from its own global DOF block — the faithful,
            // ε̃-sourced membrane part of STRAIN / TRACTION. Read it from the FULL solution.
            if (element_->field_aux_contribution(ev, pt, field_, N_aux, aux_dofs)
                && !aux_dofs.empty()) {
                const Index max_idx = *std::max_element(aux_dofs.begin(), aux_dofs.end());
                if (max_idx >= u_.size()) {
                    throw std::runtime_error(
                        "Function: this field is sourced from a mixed formulation's "
                        "auxiliary DOFs; pass the full solution (ck.solve(problem, "
                        "full=True)).");
                }
                Vector<T> u_aux(static_cast<Index>(aux_dofs.size()));
                for (Index a = 0; a < u_aux.size(); ++a)
                    u_aux(a) = u_(aux_dofs[static_cast<std::size_t>(a)]);
                out.row(q).noalias() += (N_aux * u_aux).transpose();
            }
        }
        return out;
    }

    const Vector<T>&     u()       const { return u_; }
    const Element<T, d>& element() const { return *element_; }
    const Patch<T, d>&   patch()   const { return *patch_; }
    FieldType            field()   const { return field_; }

    Vector<T>          u_;
    Ptr<Element<T, d>> element_;
    Ptr<Patch<T, d>>   patch_;
    FieldType          field_;
};

} // namespace pyck

#endif // PYCK_FUNCTION_HPP
