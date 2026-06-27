#include "penalty_coupling_condition.hpp"

#include "boundary_element_values.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pyck
{

namespace {

/// @brief Cross-block sign for a coupled field's continuity penalty.
///
/// The seam combination is set by how the trace depends on the outward boundary
/// frame, which flips between the two sides (n_A = −n_B, s_A = −s_B):
///   - frame-independent / global-axis traces (`U_X/Y/Z`, `VB_*`, `ROT_X/Y/Z`)
///     measure the same quantity on both sides → jump is a difference (sign −1);
///   - traces ODD in the frame (`U_N/U_S`, `ROT_N/ROT_S`, `PSI_N`, the director
///     rotation `DIR_N` = δa_3·n) flip sign across the seam → tied as a sum (+1);
///   - traces EVEN in the frame (the normal curvature `KAPPA_NN` = n^α n^β κ_αβ,
///     where the co-normal enters squared) do not flip → difference (−1).
template <std::floating_point T>
T coupling_sign(Field f)
{
    switch (f) {
        case Field::U_N: case Field::U_S:
        case Field::ROT_N: case Field::ROT_S:
        case Field::PSI_N: case Field::DIR_N:
            return T(1);
        default:   // global axes, VB_*, PSI, and the even-in-n KAPPA_NN
            return T(-1);
    }
}

} // namespace

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>::PenaltyCouplingCondition(
        const PatchBoundary<T, d>& boundary_a,
        const PatchBoundary<T, d>& boundary_b,
        const Element<T, d>& element,
        const QuadratureRule<T, d - 1>& quadrature)
    : boundary_a_(boundary_a),
      boundary_b_(boundary_b),
      element_(element),
      quadrature_(quadrature)
{
    // --- Orientation from physical control-net endpoints --------------------------
    // Clamped knot vectors interpolate their end control points, so the first/last
    // control points are the physical curve ends. The two sides may run the shared
    // curve in the same or opposite parametric direction.
    const ColMatrix<T, 3>& cp_a = boundary_a_.control_pts();
    const ColMatrix<T, 3>& cp_b = boundary_b_.control_pts();
    const Index na = static_cast<Index>(cp_a.rows());
    const Index nb = static_cast<Index>(cp_b.rows());

    const T edge_len = (cp_a.row(0) - cp_a.row(na - 1)).norm();
    const T geo_tol  = T(1e-9) * std::max(edge_len, T(1));
    coincidence_tol_ = T(1e-7) * std::max(edge_len, T(1));

    const bool fwd = (cp_a.row(0) - cp_b.row(0)).norm() < geo_tol
                  && (cp_a.row(na - 1) - cp_b.row(nb - 1)).norm() < geo_tol;
    const bool rev = (cp_a.row(0) - cp_b.row(nb - 1)).norm() < geo_tol
                  && (cp_a.row(na - 1) - cp_b.row(0)).norm() < geo_tol;
    if (fwd)      reversed_ = false;
    else if (rev) reversed_ = true;
    else throw std::invalid_argument(
        "PenaltyCouplingCondition: interface boundary endpoints are not physically "
        "coincident (neither forward nor reversed pairing matches).");

    // --- Affine-validity + geometric-coincidence check ----------------------------
    // The A↔B point map is the affine arc-fraction map; it is exact iff both sides
    // parametrise the shared curve identically up to that affine relation (true for
    // straight / Greville-parametrised seams). Sampling the two curves at coincident
    // affine fractions validates both that they are the same curve and that the
    // affine map holds; a curved/mismatched seam fails here and is rejected.
    constexpr Index M = 8;
    ColMatrix<T, d - 1> a_samples(M, static_cast<Index>(d - 1));
    const T a_lo = boundary_a_.basis(0).knot_vector().front();
    const T a_hi = boundary_a_.basis(0).knot_vector().back();
    for (Index k = 0; k < M; ++k) {
        const T frac = T(k + 1) / T(M + 1);
        a_samples(k, 0) = a_lo + frac * (a_hi - a_lo);
    }
    const ColMatrix<T, 3> x_a = boundary_a_.eval_geometry(a_samples);
    const ColMatrix<T, 3> x_b = boundary_b_.eval_geometry(map_a_to_b(a_samples));
    for (Index k = 0; k < M; ++k) {
        if ((x_a.row(k) - x_b.row(k)).norm() > coincidence_tol_) {
            throw std::invalid_argument(
                "PenaltyCouplingCondition: interface geometry / parametrisation "
                "mismatch — the affine arc-fraction map is invalid. A curved or "
                "mismatched-parametrisation seam needs point inversion "
                "(not yet supported).");
        }
    }

    // --- Knot-span index -> live boundary-span index, per side --------------------
    auto build_inverse = [](const PatchBoundary<T, d>& bnd, std::vector<Index>& inv) {
        const auto& kv = bnd.basis(0).knot_vector();
        inv.assign(static_cast<std::size_t>(kv.num_spans()), Index(0));
        const auto& nz = kv.non_zero_spans();
        for (std::size_t live = 0; live < nz.size(); ++live) {
            inv[static_cast<std::size_t>(nz[live])] = static_cast<Index>(live);
        }
    };
    build_inverse(boundary_a_, a_span_inv_);
    build_inverse(boundary_b_, b_span_inv_);

    // --- Common-refinement segmentation -------------------------------------------
    build_segments();
}

// === Affine arc-fraction maps =======================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
T PenaltyCouplingCondition<T, d>::map_a_to_b(T a_param) const
{
    const T a_lo = boundary_a_.basis(0).knot_vector().front();
    const T a_hi = boundary_a_.basis(0).knot_vector().back();
    const T b_lo = boundary_b_.basis(0).knot_vector().front();
    const T b_hi = boundary_b_.basis(0).knot_vector().back();
    const T t = (a_param - a_lo) / (a_hi - a_lo);
    const T tb = reversed_ ? (T(1) - t) : t;
    return b_lo + tb * (b_hi - b_lo);
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
T PenaltyCouplingCondition<T, d>::map_b_to_a(T b_param) const
{
    const T a_lo = boundary_a_.basis(0).knot_vector().front();
    const T a_hi = boundary_a_.basis(0).knot_vector().back();
    const T b_lo = boundary_b_.basis(0).knot_vector().front();
    const T b_hi = boundary_b_.basis(0).knot_vector().back();
    const T t = (b_param - b_lo) / (b_hi - b_lo);
    const T ta = reversed_ ? (T(1) - t) : t;
    return a_lo + ta * (a_hi - a_lo);
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
ColMatrix<T, d - 1>
PenaltyCouplingCondition<T, d>::map_a_to_b(const ColMatrix<T, d - 1>& a_pts) const
{
    ColMatrix<T, d - 1> b_pts(a_pts.rows(), static_cast<Index>(d - 1));
    for (Index q = 0; q < static_cast<Index>(a_pts.rows()); ++q) {
        b_pts(q, 0) = map_a_to_b(a_pts(q, 0));
    }
    return b_pts;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
Index PenaltyCouplingCondition<T, d>::a_live_span(T a_param) const
{
    return a_span_inv_[static_cast<std::size_t>(boundary_a_.basis(0).find_span(a_param))];
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
Index PenaltyCouplingCondition<T, d>::b_live_span(T b_param) const
{
    return b_span_inv_[static_cast<std::size_t>(boundary_b_.basis(0).find_span(b_param))];
}

// === Segmentation ===================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
void PenaltyCouplingCondition<T, d>::build_segments()
{
    const auto& kv_a = boundary_a_.basis(0).knot_vector();
    const auto& kv_b = boundary_b_.basis(0).knot_vector();
    const T a_lo = kv_a.front();
    const T a_hi = kv_a.back();
    const T merge_tol = T(1e-10) * (a_hi - a_lo);

    // Breakpoints in A-parameter: A's live-span bounds plus the A-images of B's
    // live-span bounds. Their union is the common refinement of the two meshes.
    std::vector<T> breaks;
    for (int s : kv_a.non_zero_spans()) {
        auto [lo, hi] = kv_a.span_bounds(s);
        breaks.push_back(lo);
        breaks.push_back(hi);
    }
    for (int s : kv_b.non_zero_spans()) {
        auto [lo, hi] = kv_b.span_bounds(s);
        breaks.push_back(map_b_to_a(lo));
        breaks.push_back(map_b_to_a(hi));
    }
    for (T& x : breaks) x = std::clamp(x, a_lo, a_hi);
    std::sort(breaks.begin(), breaks.end());

    // Dedup near-coincident breakpoints (collapses conforming breakpoints exactly
    // and kills slivers where an A and a mapped-B breakpoint nearly meet).
    std::vector<T> nodes;
    nodes.reserve(breaks.size());
    for (T x : breaks) {
        if (nodes.empty() || x - nodes.back() > merge_tol) nodes.push_back(x);
    }

    const Index n_gp = static_cast<Index>(quadrature_.num_points());
    for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
        const T x0 = nodes[i];
        const T x1 = nodes[i + 1];
        if (x1 - x0 <= merge_tol) continue;   // sliver guard

        const T mid = T(0.5) * (x0 + x1);

        Segment seg;
        seg.a_span_live = a_live_span(mid);
        seg.b_span_live = b_live_span(map_a_to_b(mid));
        seg.a_pts.resize(n_gp, static_cast<Index>(d - 1));
        seg.weights.resize(n_gp);
        quadrature_.map_to_domain(std::array<T, d - 1>{x0}, std::array<T, d - 1>{x1},
                                  seg.a_pts, seg.weights);
        seg.b_pts = map_a_to_b(seg.a_pts);
        segments_.push_back(std::move(seg));
    }
}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>&
PenaltyCouplingCondition<T, d>::add(BoundaryValue<T> field, T penalty)
{
    const Field f = field.field();
    if (f != Field::U_X && f != Field::U_Y && f != Field::U_Z
        && f != Field::VB_X && f != Field::VB_Y && f != Field::VB_Z
        && f != Field::ROT_N && f != Field::DIR_N && f != Field::KAPPA_NN
        && f != Field::PSI && f != Field::PSI_N) {
        throw std::invalid_argument(
            "PenaltyCouplingCondition::add: supported fields are the total displacement "
            "(U_X, U_Y, U_Z), the bending-surface displacement (VB_X, VB_Y, VB_Z), the "
            "normal/bending rotation (ROT_N), the director-based normal rotation (DIR_N, "
            "C1) and normal curvature (KAPPA_NN, C2), and the hierarchic shear potential "
            "and its normal slope (PSI, PSI_N) of the four-parameter shells. Other "
            "frame/rotation components are not validated for coupling.");
    }
    terms_.push_back({std::move(field), penalty});
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>&
PenaltyCouplingCondition<T, d>::couple_kinematics(T penalty_disp, T penalty_rot)
{
    add(BoundaryValue<T>(Field::VB_X), penalty_disp);
    add(BoundaryValue<T>(Field::VB_Y), penalty_disp);
    add(BoundaryValue<T>(Field::VB_Z), penalty_disp);
    add(BoundaryValue<T>(Field::ROT_N), penalty_rot);
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>&
PenaltyCouplingCondition<T, d>::couple_displacement(T penalty)
{
    add(BoundaryValue<T>(Field::VB_X), penalty);
    add(BoundaryValue<T>(Field::VB_Y), penalty);
    add(BoundaryValue<T>(Field::VB_Z), penalty);
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>&
PenaltyCouplingCondition<T, d>::couple_director_continuity(T penalty)
{
    // C1 slope continuity: the director-based normal rotation δa_3·n, per patch, summed
    // across the seam (odd in the co-normal, so n_A = −n_B flips it).
    add(BoundaryValue<T>(Field::DIR_N), penalty);
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>&
PenaltyCouplingCondition<T, d>::couple_curvature_continuity(T penalty)
{
    // C2 curvature continuity: the normal curvature n^α n^β κ_αβ, per patch, differenced
    // across the seam (even in the co-normal, so n_A = −n_B leaves it unchanged).
    add(BoundaryValue<T>(Field::KAPPA_NN), penalty);
    return *this;
}

template <std::floating_point T, std::size_t d>
requires (d > 1)
PenaltyCouplingCondition<T, d>&
PenaltyCouplingCondition<T, d>::couple_psi_continuity(T penalty_value, T penalty_slope)
{
    add(BoundaryValue<T>(Field::PSI),   penalty_value);
    add(BoundaryValue<T>(Field::PSI_N), penalty_slope);
    return *this;
}

// === Assembly =======================================================================

template <std::floating_point T, std::size_t d>
requires (d > 1)
void PenaltyCouplingCondition<T, d>::apply(SystemAssembler<T>& assembler,
                                           const DofLayout& layout,
                                           const PatchBlocks<T, d>& blocks) const
{
    if (terms_.empty()) return;

    const DofLayout::BlockId block_a = blocks.primal(patch_a());
    const DofLayout::BlockId block_b = blocks.primal(patch_b());

    // Infer the basis order and quantity flags needed by the element + fields.
    Index    order = element_.basis_order();
    unsigned flags = Flags::None;
    for (const auto& term : terms_) {
        order = std::max(order, term.field.basis_order());
        flags |= term.field.flags();
        flags |= term.field.element_flags(element_);
    }

    // Two-basis workspaces, one per side; both evaluated at the segment's points.
    BoundaryElementValues<T, d> bd_a(boundary_a_, order, flags, quadrature_);
    BoundaryElementValues<T, d> bd_b(boundary_b_, order, flags, quadrature_);

    const Index ndof = static_cast<Index>(element_.num_node_dofs());
    const Index n_a  = static_cast<Index>(bd_a.parent_vals_.act_pts_.rows()) * ndof;
    const Index n_b  = static_cast<Index>(bd_b.parent_vals_.act_pts_.rows()) * ndof;
    const Index n_gp = static_cast<Index>(quadrature_.num_points());

    // --- Local coupling blocks ------------------------------------------------------
    // K = α ∫ [N_A; −N_B]ᵀ [N_A; −N_B] dΓ  →  K_AA, K_AB, K_BA, K_BB, reused per segment.
    Matrix<T> K_AA(n_a, n_a), K_AB(n_a, n_b), K_BA(n_b, n_a), K_BB(n_b, n_b);

    for (const auto& seg : segments_) {
        // Side A self-drives the measure; side B at the affinely-coincident points.
        bd_a.reinit_on_boundary_pts(seg.a_span_live, seg.a_pts);
        bd_b.reinit_on_boundary_pts(seg.b_span_live, seg.b_pts);

#ifndef NDEBUG
        // Guard: the paired stations must be physically coincident. A failure means
        // the segmentation / affine map is wrong, which would assemble silent garbage.
        for (Index q = 0; q < n_gp; ++q) {
            const Vector3<T> xa = bd_a.parent_vals_.position(q);
            const Vector3<T> xb = bd_b.parent_vals_.position(q);
            assert((xa - xb).norm() <= coincidence_tol_
                   && "PenaltyCouplingCondition: paired interface stations are not "
                      "physically coincident");
        }
#endif

        // Global DOFs for both sides on this segment.
        layout.scatter_primal(block_a, bd_a.parent_vals_.elem_cps_,
                              bd_a.parent_vals_.elem_dofs_);
        layout.scatter_primal(block_b, bd_b.parent_vals_.elem_cps_,
                              bd_b.parent_vals_.elem_dofs_);
        const auto& dof_a = bd_a.parent_vals_.elem_dofs_;
        const auto& dof_b = bd_b.parent_vals_.elem_dofs_;

        K_AA.setZero(); K_AB.setZero(); K_BA.setZero(); K_BB.setZero();

        for (const auto& term : terms_) {
            const Matrix<T> N_A = term.field.evaluate(element_, bd_a);  // (n_gp × n_a)
            const Matrix<T> N_B = term.field.evaluate(element_, bd_b);  // (n_gp × n_b)
            // Difference (−1) for global axes, sum (+1) for frame projections.
            const T sgn = coupling_sign<T>(term.field.field());

            for (Index q = 0; q < n_gp; ++q) {
                // Arc-length measure from side A; segment weights carry the interval
                // scaling (reinit_on_boundary_pts does NOT populate mapped_weights_).
                const T ds = bd_a.boundary_vals_.jac(q) * seg.weights(q);
                const T w = term.penalty * ds;
                const auto r_a = N_A.row(q);
                const auto r_b = N_B.row(q);
                // jump² with positive diagonals; off-diagonals carry sgn:
                //   global  (w_A − w_B): K_AB = −α NᵀA N_B
                //   frame   (w_A + w_B): K_AB = +α NᵀA N_B
                K_AA.noalias() += w * (r_a.transpose() * r_a);
                K_AB.noalias() += sgn * w * (r_a.transpose() * r_b);
                K_BA.noalias() += sgn * w * (r_b.transpose() * r_a);
                K_BB.noalias() += w * (r_b.transpose() * r_b);
            }
        }

        // --- Scatter the four blocks ------------------------------------------------
        for (Index i = 0; i < n_a; ++i) {
            for (Index j = 0; j < n_a; ++j)
                assembler.add_stiffness(dof_a[i], dof_a[j], K_AA(i, j));
            for (Index j = 0; j < n_b; ++j)
                assembler.add_stiffness(dof_a[i], dof_b[j], K_AB(i, j));
        }
        for (Index i = 0; i < n_b; ++i) {
            for (Index j = 0; j < n_a; ++j)
                assembler.add_stiffness(dof_b[i], dof_a[j], K_BA(i, j));
            for (Index j = 0; j < n_b; ++j)
                assembler.add_stiffness(dof_b[i], dof_b[j], K_BB(i, j));
        }
    }
}

// === Template Instantiations ========================================================

template class PenaltyCouplingCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class PenaltyCouplingCondition<float, 2>;
#endif

} // namespace pyck
