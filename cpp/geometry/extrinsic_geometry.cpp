#include "extrinsic_geometry.hpp"

#include <Eigen/Core>
#include <cassert>

namespace pyck
{

template <std::floating_point T, std::size_t d>
void ExtrinsicGeometry<T, d>::reinit(const IntrinsicGeometry<T, d>& ig,
                                     ExtrinsicGeometryFlags flags)
{
    if constexpr (d != 2) {
        // Extrinsic geometry as defined here (surface normal + curvature in 3D
        // ambient space) is only meaningful for d == 2. Leave storage empty.
        (void)ig;
        (void)flags;
        return;
    } else {
        const Index Q_ = ig.Q();

        // Best-effort: clamp flags to what the source intrinsic geometry can
        // provide. Callers using default flags get the maximum available.
        if (ig.order() < 2) {
            flags.normal_derivatives = false;
            flags.curvature          = false;
        }

        // --- Allocations (flag-gated; no-op when already sized) -----------------------

        n.resize(Q_, 3);
        if (flags.normal_derivatives) n_d1_data.resize(Q_ * 2, 3);
        if (flags.curvature) {
            b_data.resize(Q_, 3);          // n_metric = 3 for d=2
            b_mixed_data.resize(Q_, 4);    // 2 × 2 full
        }

        // --- Pass 1: unit normal (always) ---------------------------------------------

        for (Index q = 0; q < Q_; ++q)
        {
            Eigen::Matrix<T, 3, 1> v =
                ig.a(0).row(q).transpose().cross(ig.a(1).row(q).transpose());
            const T J = ig.jac(q);
            if (J > T(1e-14)) v /= J;
            else              v = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
            n.row(q) = v.transpose();
        }

        // --- Pass 2: normal derivatives ----------------------------------------------

        if (flags.normal_derivatives)
        {
            for (Index q = 0; q < Q_; ++q)
            {
                const Eigen::Matrix<T, 3, 1> a0  = ig.a(0).row(q).transpose();
                const Eigen::Matrix<T, 3, 1> a1  = ig.a(1).row(q).transpose();
                const Eigen::Matrix<T, 3, 1> a3  = n.row(q).transpose();
                const Eigen::Matrix<T, 3, 1> a00 = ig.a_d1(0, 0).row(q).transpose();
                const Eigen::Matrix<T, 3, 1> a01 = ig.a_d1(0, 1).row(q).transpose();
                const Eigen::Matrix<T, 3, 1> a11 = ig.a_d1(1, 1).row(q).transpose();

                const T inv_J = (ig.jac(q) > T(1e-14)) ? T(1) / ig.jac(q) : T(0);

                // ∂_β N where N = a_0 × a_1, β ∈ {0, 1}.
                const Eigen::Matrix<T, 3, 1> dN0 = a00.cross(a1) + a0.cross(a01);
                const Eigen::Matrix<T, 3, 1> dN1 = a01.cross(a1) + a0.cross(a11);

                // Tangent-plane projection: ∂_β a_3 = [(I - a_3 a_3ᵀ) ∂_β N] / J.
                n_d1_data.middleRows(0 * Q_, Q_).row(q) =
                    ((dN0 - a3.dot(dN0) * a3) * inv_J).transpose();
                n_d1_data.middleRows(1 * Q_, Q_).row(q) =
                    ((dN1 - a3.dot(dN1) * a3) * inv_J).transpose();
            }
        }

        // --- Pass 3: curvature (second fundamental form + shape operator) ------------

        if (flags.curvature)
        {
            for (Index q = 0; q < Q_; ++q)
            {
                // b_{αβ} = a_{αβ} · a_3   (sym, α ≤ β filled via pack2)
                const T b11 = ig.a_d1(0, 0).row(q).dot(n.row(q));
                const T b12 = ig.a_d1(0, 1).row(q).dot(n.row(q));
                const T b22 = ig.a_d1(1, 1).row(q).dot(n.row(q));
                b_data(q, pack2<2>(0, 0)) = b11;
                b_data(q, pack2<2>(0, 1)) = b12;
                b_data(q, pack2<2>(1, 1)) = b22;

                // b^α_β = g^{αγ} b_{γβ}   (4 entries, not symmetric)
                const T gi11 = ig.g_inv(0, 0)(q);
                const T gi12 = ig.g_inv(0, 1)(q);
                const T gi22 = ig.g_inv(1, 1)(q);
                b_mixed_data(q, 0 * 2 + 0) = gi11 * b11 + gi12 * b12;
                b_mixed_data(q, 0 * 2 + 1) = gi11 * b12 + gi12 * b22;
                b_mixed_data(q, 1 * 2 + 0) = gi12 * b11 + gi22 * b12;
                b_mixed_data(q, 1 * 2 + 1) = gi12 * b12 + gi22 * b22;
            }
        }
    }
}

// === Template Instantiations ========================================================

template class ExtrinsicGeometry<double, 1>;
template class ExtrinsicGeometry<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class ExtrinsicGeometry<float, 1>;
template class ExtrinsicGeometry<float, 2>;
#endif

} // namespace pyck
