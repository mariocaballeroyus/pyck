#ifndef PYCK_FIELD_TRANSFORM_HPP
#define PYCK_FIELD_TRANSFORM_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>

#include "../elements/element_values.hpp"
#include "../geometry/patch.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Map covariant vector components to physical Cartesian components.
 *
 * @details A surface field given by its covariant components @f$ v_\alpha = v\cdot A_\alpha @f$
 *          (e.g. the rotation @f$ \theta_\alpha @f$, transverse shear @f$ \gamma_\alpha @f$,
 *          or the curl-of-ψ shear) is *basis dependent*: across a non-conforming or
 *          slanted multipatch seam the local tangent basis @f$ A_\alpha @f$ differs in
 *          length and direction, so a covariant component looks discontinuous even when
 *          the physical vector is continuous. This raises the index and re-expresses the
 *          field in the global Cartesian frame,
 *          @f$ v = v_\alpha A^\alpha = v_\alpha A^{\alpha\beta} A_\beta @f$, giving
 *          frame-consistent components that are continuous across the seam — the form to
 *          export for visualization.
 *
 * @param patch  Surface patch carrying the geometry.
 * @param params (Q × 2) parametric evaluation points.
 * @param comps  (Q × 2) covariant components @f$ v_\alpha @f$ at those points.
 * @return       (Q × 3) physical Cartesian components.
 */
template <std::floating_point T>
Matrix<T> covariant_to_cartesian(const Patch<T, 2>& patch,
                                 const ColMatrix<T, 2>& params,
                                 const Matrix<T>& comps)
{
    const Index Q = static_cast<Index>(params.rows());
    if (static_cast<Index>(comps.rows()) != Q || comps.cols() != 2)
        throw std::runtime_error(
            "covariant_to_cartesian: comps must be (Q x 2) matching params (Q x 2).");

    // Geometry-only workspace: the covariant basis A_α and inverse metric A^{αβ}.
    ElementValues<T, 2> ev(patch, Index(1), Flags::Deriv1, std::size_t(1));

    Matrix<T> out(Q, 3);
    for (Index q = 0; q < Q; ++q) {
        ColMatrix<T, 2> pt = params.row(q);
        std::array<Index, 2> span_idx;
        for (std::size_t dir = 0; dir < 2; ++dir)
            span_idx[dir] = patch.basis(dir).find_span(pt(0, static_cast<Index>(dir)));
        ev.reinit_on_pts(span_idx, pt);

        const auto A = ev.cov_basis(0);
        const Vector3<T> A1 = A(0), A2 = A(1);
        const Eigen::Matrix<T, 2, 2> gi = ev.metric_inv(0);

        // Contravariant basis A^α = A^{αβ} A_β, then v = v_α A^α.
        const Vector3<T> Aup1 = gi(0, 0) * A1 + gi(0, 1) * A2;
        const Vector3<T> Aup2 = gi(1, 0) * A1 + gi(1, 1) * A2;
        out.row(q) = (comps(q, 0) * Aup1 + comps(q, 1) * Aup2).transpose();
    }
    return out;
}

} // namespace pyck

#endif // PYCK_FIELD_TRANSFORM_HPP
