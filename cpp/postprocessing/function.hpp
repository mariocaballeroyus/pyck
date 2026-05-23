#ifndef PYCK_FUNCTION_HPP
#define PYCK_FUNCTION_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>

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
        const Index expected = static_cast<Index>(patch_->num_control_pts())
                             * static_cast<Index>(element_->num_node_dofs());
                             
        if (u_.size() != expected) {
            throw std::runtime_error(
                "Function: DOF vector has " + std::to_string(u_.size()) +
                " entries but patch/element imply " + std::to_string(expected) +
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
                case FieldType::DISPLACEMENT: e.displacement_shape_matrix(ev); return e.N_w_workspace_;
                case FieldType::ROTATION:     e.rotation_shape_matrix(ev);     return e.N_phi_workspace_;
                case FieldType::STRAIN:       e.strain_matrix(ev);             return e.B_workspace_;
                case FieldType::STRESS:       e.stress_matrix(ev);             return e.N_sigma_workspace_;
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
