#include "mixed_formulation.hpp"

#include <stdexcept>
#include <utility>

#include "../elements/element_values.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
MixedFormulation<T, d>::MixedFormulation(Ptr<Patch<T, d>>          patch,
                                         Ptr<Element<T, d>>        element,
                                         Ptr<QuadratureRule<T, d>> quadrature)
    : patch_(std::move(patch)), element_(std::move(element)),
      quadrature_(std::move(quadrature))
{
    if (!patch_ || !element_ || !quadrature_) {
        throw std::invalid_argument(
            "MixedFormulation: null patch, element, or quadrature.");
    }
}

template <std::floating_point T, std::size_t d>
void
MixedFormulation<T, d>::allocate_dofs(DofLayout& layout)
{
    allocate_field_dofs(layout);
    // The displacement element no longer assembles the governed block's
    // stiffness; this formulation supplies it through the saddle below.
    element_->suppress_block(governed_block());
}

template <std::floating_point T, std::size_t d>
void
MixedFormulation<T, d>::apply(SystemAssembler<T>&      assembler,
                              const DofLayout&         layout,
                              const PatchBlocks<T, d>& blocks) const
{
    const DofLayout::BlockId primal_block = blocks.primal(*patch_);
    const StrainBlock        blk  = governed_block();
    const Index              ndof = static_cast<Index>(element_->num_node_dofs());

    ElementValues<T, d> ev(*patch_, element_->basis_order(), element_->flags(),
                           *quadrature_);
    const Index n_elem = ev.num_elements();

    Matrix<T>          O;
    std::vector<Index> u_dofs, e_dofs;

    for (Index e = 0; e < n_elem; ++e) {
        ev.reinit(e);
        element_->strain_matrix(ev);
        const Matrix<T>& B  = element_->B_voigt_;
        const Index      Q  = ev.num_points();
        const Index      ns = static_cast<Index>(B.rows()) / Q;   // strain rows per qp
        const Index      N  = static_cast<Index>(ev.basis_derivs[0].rows());
        const Index      nu = ndof * N;

        field_contribution(ev, e, layout, O, e_dofs);
        const Index ne = static_cast<Index>(O.cols());

        Matrix<T> G = Matrix<T>::Zero(ne, nu);   // coupling: field rows × displacement cols
        Matrix<T> H = Matrix<T>::Zero(ne, ne);   // field compliance

        for (Index q = 0; q < Q; ++q) {
            const T dV = ev.mapped_weights_(q) * ev.jac(q);
            const ConstitutiveMatrix<T> D = element_->constitutive_matrix(ev, q);
            const Matrix<T> Db = D.block(blk.offset, blk.offset, blk.count, blk.count);
            const auto      Bb = B.middleRows(ns * q + blk.offset, blk.count);  // count × nu
            const auto      Oq = O.middleRows(blk.count * q, blk.count);        // count × ne
            const Matrix<T> DB = Db * Bb;                                        // count × nu
            const Matrix<T> DO = Db * Oq;                                        // count × ne
            G.noalias() += dV * (Oq.transpose() * DB);
            H.noalias() += dV * (Oq.transpose() * DO);
        }

        layout.scatter_primal(primal_block, ev.elem_cps_, u_dofs);   // nu global indices

        for (Index a = 0; a < ne; ++a) {
            for (Index j = 0; j < nu; ++j) {
                const T g = G(a, j);
                assembler.add_stiffness(e_dofs[a], u_dofs[j], g);
                assembler.add_stiffness(u_dofs[j], e_dofs[a], g);
            }
            for (Index b = 0; b < ne; ++b)
                assembler.add_stiffness(e_dofs[a], e_dofs[b], -H(a, b));
        }
    }
}

// === Template Instantiations ========================================================

template class MixedFormulation<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class MixedFormulation<float, 2>;
#endif

} // namespace pyck
