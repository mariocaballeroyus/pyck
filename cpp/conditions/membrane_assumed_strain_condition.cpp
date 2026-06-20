#include "membrane_assumed_strain_condition.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../basis/bspline.hpp"
#include "../basis/knot_vector.hpp"
#include "../elements/element_values.hpp"

namespace pyck
{

namespace
{

/// @brief Build a B-spline of degree (p - drop) over the displacement breakpoints with
///        maximal continuity (interior multiplicity 1) — the coarser assumed-strain space.
template <std::floating_point T>
Ptr<const Basis<T>> make_reduced_bspline(const Basis<T>& disp, Index drop)
{
    const Index p  = disp.degree();
    const Index rp = std::max<Index>(Index(1), p - drop);

    // Unique breakpoints of the displacement basis (the element partition).
    std::vector<T> bp;
    for (T k : disp.knots())
        if (bp.empty() || k > bp.back()) bp.push_back(k);

    // Clamped knot vector: ends repeated (rp+1) times, interior breakpoints once.
    std::vector<T> kv;
    kv.reserve(bp.size() + 2 * static_cast<std::size_t>(rp));
    for (Index r = 0; r <= rp; ++r) kv.push_back(bp.front());
    for (std::size_t i = 1; i + 1 < bp.size(); ++i) kv.push_back(bp[i]);
    for (Index r = 0; r <= rp; ++r) kv.push_back(bp.back());

    return std::make_shared<BSpline<T>>(rp, KnotVector<T>(std::move(kv)));
}

} // namespace

template <std::floating_point T>
MembraneAssumedStrainCondition<T>::MembraneAssumedStrainCondition(
    Ptr<Patch<T, 2>> patch, Ptr<Element<T, 2>> element,
    Ptr<QuadratureRule<T, 2>> quadrature, Index degree_drop)
    : patch_(std::move(patch)), element_(std::move(element)),
      quadrature_(std::move(quadrature)), degree_drop_(degree_drop)
{
    if (!patch_ || !element_ || !quadrature_) {
        throw std::invalid_argument(
            "MembraneAssumedStrainCondition: null patch, element, or quadrature.");
    }

    const Basis<T>& b0 = patch_->basis(0);
    const Basis<T>& b1 = patch_->basis(1);

    const Ptr<const Basis<T>> d0_full = make_reduced_bspline<T>(b0, 0);
    const Ptr<const Basis<T>> d0_red  = make_reduced_bspline<T>(b0, degree_drop_);
    const Ptr<const Basis<T>> d1_full = make_reduced_bspline<T>(b1, 0);
    const Ptr<const Basis<T>> d1_red  = make_reduced_bspline<T>(b1, degree_drop_);

    // Echter Table 6.2 / Guo Eq. 29: ε̃₁₁ ~ (p-1,q), ε̃₂₂ ~ (p,q-1), 2ε̃₁₂ ~ (p-1,q-1).
    comp_basis_[0] = std::make_shared<TensorProduct<T, 2>>(
        std::array<Ptr<const Basis<T>>, 2>{d0_red, d1_full});
    comp_basis_[1] = std::make_shared<TensorProduct<T, 2>>(
        std::array<Ptr<const Basis<T>>, 2>{d0_full, d1_red});
    comp_basis_[2] = std::make_shared<TensorProduct<T, 2>>(
        std::array<Ptr<const Basis<T>>, 2>{d0_red, d1_red});

    for (int c = 0; c < 3; ++c) {
        const TensorProduct<T, 2>& tp = *comp_basis_[c];
        const std::array<Index, 2> nb { tp.basis(0).num_basis(), tp.basis(1).num_basis() };
        const std::array<Index, 2> dg { tp.basis(0).degree(),    tp.basis(1).degree() };
        comp_mapper_[c].init(nb, dg);
        comp_ncp_[c] = nb[0] * nb[1];
    }
}

template <std::floating_point T>
void
MembraneAssumedStrainCondition<T>::allocate_dofs(DofLayout& layout)
{
    for (int c = 0; c < 3; ++c) {
        comp_block_[c] = layout.allocate(DofType::LagrangeMultiplier, comp_ncp_[c], 1);
        comp_base_[c]  = layout.block_base(comp_block_[c]);   // for membrane-force recovery
    }
    // The displacement element no longer assembles the membrane block's stiffness; this
    // condition re-supplies it through the saddle below (and recovers its stress).
    element_->suppress_block({0, 3});
}

template <std::floating_point T>
void
MembraneAssumedStrainCondition<T>::field_contribution(
    const ElementValues<T, 2>& ev, Index e, const DofLayout& layout,
    Matrix<T>& O, std::vector<Index>& dofs) const
{
    const Index Q = ev.num_points();
    const std::array<Index, 3> ebase {
        layout.block_base(comp_block_[0]),
        layout.block_base(comp_block_[1]),
        layout.block_base(comp_block_[2]) };

    std::array<std::vector<Matrix<T>>, 2> uni;     // eval_on_span scratch
    std::array<std::vector<Matrix<T>>, 3> Rres;    // reduced basis values per component
    std::array<std::vector<Index>, 3>     e_cps;
    std::array<Index, 3>                  Kc;

    for (int c = 0; c < 3; ++c) {
        const std::array<Index, 2> spans = comp_basis_[c]->decode_element(e);
        comp_basis_[c]->eval_on_span(ev.mapped_pts_, spans, 0, uni, Rres[c]);
        Kc[c] = static_cast<Index>(Rres[c][0].rows());
        comp_mapper_[c].get_element_cps(spans, e_cps[c]);
    }
    const Index ne = Kc[0] + Kc[1] + Kc[2];

    // Stacked field operator (3·Q × ne): row block at qp q holds, per component c,
    // that component's reduced-basis values in its column range (the R̃ of Guo Eq. 36).
    O.setZero(3 * Q, ne);
    for (Index q = 0; q < Q; ++q) {
        Index off = 0;
        for (int c = 0; c < 3; ++c) {
            for (Index k = 0; k < Kc[c]; ++k)
                O(3 * q + c, off + k) = Rres[c][0](k, q);
            off += Kc[c];
        }
    }

    dofs.clear();
    for (int c = 0; c < 3; ++c)
        for (Index k = 0; k < Kc[c]; ++k)
            dofs.push_back(ebase[c] + e_cps[c][k]);
}

template <std::floating_point T>
void
MembraneAssumedStrainCondition<T>::apply(SystemAssembler<T>&      assembler,
                                         const DofLayout&         layout,
                                         const PatchBlocks<T, 2>& blocks) const
{
    const DofLayout::BlockId primal_block = blocks.primal(*patch_);
    const StrainBlock        blk{0, 3};   // membrane rows (ε₁₁, ε₂₂, 2ε₁₂)
    const Index              ndof = static_cast<Index>(element_->num_node_dofs());

    ElementValues<T, 2> ev(*patch_, element_->basis_order(), element_->flags(),
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

template <std::floating_point T>
Matrix<T>
MembraneAssumedStrainCondition<T>::recover_membrane_force(
    const Vector<T>& full_u, const ColMatrix<T, 2>& params) const
{
    const Index Q = static_cast<Index>(params.rows());
    Matrix<T> out(Q, 3);

    // Single-point workspace for the curvilinear membrane constitutive D_m per point.
    ElementValues<T, 2> ev(*patch_, element_->basis_order(),
                           element_->flags(), std::size_t(1));

    std::array<std::vector<Matrix<T>>, 2> uni;     // eval_on_span scratch
    std::array<std::vector<Matrix<T>>, 3> Rres;
    std::array<std::vector<Index>, 3>     cps;

    for (Index q = 0; q < Q; ++q) {
        const ColMatrix<T, 2> pt = params.row(q);

        // Membrane constitutive D_m at the point (metric from the geometry).
        std::array<Index, 2> span_idx;
        for (std::size_t dir = 0; dir < 2; ++dir)
            span_idx[dir] = patch_->basis(dir).find_span(pt(0, static_cast<Index>(dir)));
        ev.reinit_on_pts(span_idx, pt);
        const ConstitutiveMatrix<T> D = element_->constitutive_matrix(ev, 0);
        const Eigen::Matrix<T, 3, 3> Dm = D.template block<3, 3>(0, 0);

        // Assumed strain ε̃ = [ε̃₁₁, ε̃₂₂, 2ε̃₁₂] from the solved field coefficients.
        Eigen::Matrix<T, 3, 1> eps;
        for (int c = 0; c < 3; ++c) {
            std::array<Index, 2> spans;
            for (std::size_t dir = 0; dir < 2; ++dir)
                spans[dir] = comp_basis_[c]->basis(dir).find_span(pt(0, static_cast<Index>(dir)));
            comp_basis_[c]->eval_on_span(pt, spans, 0, uni, Rres[c]);
            comp_mapper_[c].get_element_cps(spans, cps[c]);
            const Matrix<T>& R = Rres[c][0];
            T val = T(0);
            for (Index k = 0; k < static_cast<Index>(R.rows()); ++k)
                val += R(k, 0) * full_u(comp_base_[c] + cps[c][k]);
            eps(c) = val;
        }
        out.row(q).noalias() = (Dm * eps).transpose();
    }
    return out;
}

// === Template Instantiations ========================================================

template class MembraneAssumedStrainCondition<double>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class MembraneAssumedStrainCondition<float>;
#endif

} // namespace pyck
