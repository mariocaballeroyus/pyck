#include "surface.hpp"
#include "bspline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../quadrature/quadrature.hpp"

namespace pyck
{

template <std::floating_point T>
SurfacePatch<T>::SurfacePatch(Ptr<const Basis<T>> basis_u,
                              Ptr<const Basis<T>> basis_v,
                              const ColMatrix<T, 3>& control_pts)
    : Patch<T, 2>(control_pts),
      tensor_product_(std::move(basis_u), std::move(basis_v)),
      dof_mapper_({tensor_product_.basis(0).num_basis(),
                   tensor_product_.basis(1).num_basis()},
                  {tensor_product_.basis(0).degree(),
                   tensor_product_.basis(1).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("SurfacePatch: "
                                    "Control points must be embedded in 3D space.");
    }

    const Index expected_n = tensor_product_.basis(0).num_basis()
                           * tensor_product_.basis(1).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());

    if (actual_n != expected_n) {
        throw std::invalid_argument(
            "SurfacePatch: Dimension mismatch. Expected "
            + std::to_string(expected_n) + " control points, got "
            + std::to_string(actual_n) + ".");
    }
}

template <std::floating_point T>
std::array<Index, 2> SurfacePatch<T>::decode_span(Index flat_idx) const
{
    auto intervals = tensor_product_.num_intervals();
    std::array<Index, 2> spans;
    spans[1] = flat_idx % intervals[1];
    spans[0] = flat_idx / intervals[1];
    return spans;
}

template <std::floating_point T>
std::vector<Matrix<T>> SurfacePatch<T>::eval_basis_functions(
    const ColMatrix<T, 2>& points,
    Index span,
    std::size_t order) const
{
    auto spans = decode_span(span);
    return tensor_product_.eval_on_span(points, spans, order);
}

template <std::floating_point T>
std::pair<std::vector<Matrix<T>>, Vector<T>>
SurfacePatch<T>::eval_shape_functions(
    const ColMatrix<T, 2>& points,
    Index span,
    std::size_t order) const
{
    auto spans = decode_span(span);
    order = std::max(std::size_t(1), std::min(order, std::size_t(2)));
    const Index Q = points.rows();

    // Evaluate parametric basis derivatives up to 'order'.
    // For order=2 the tensor product returns (order+1)^2 = 9 matrices.
    // Layout:  idx = du * (order+1) + dv
    //   (0,0)=N   (0,1)=N_v   (0,2)=N_vv
    //   (1,0)=N_u (1,1)=N_uv  (1,2)=N_uvv
    //   (2,0)=N_uu(2,1)=N_uuv (2,2)=N_uuvv
    auto basis_derivs = tensor_product_.eval_on_span(points, spans, order);
    const Index K = basis_derivs[0].cols();
    const Index S = order + 1;  // stride for flat index

    auto act_pts = this->active_control_pts(spans);

    // Tangent vectors  a_α = ∂x/∂ξ^α   (Q × 3 each)
    ColMatrix<T, 3> a_1 = basis_derivs[1 * S + 0] * act_pts;  // N_{,u} * P
    ColMatrix<T, 3> a_2 = basis_derivs[0 * S + 1] * act_pts;  // N_{,v} * P

    // Build output ---------------------------------------------------------
    std::vector<Matrix<T>> result;
    result.reserve(order == 1 ? 3 : 6);
    Vector<T> jacobian(Q);

    // result[0] = N  (values)
    result.push_back(basis_derivs[0]);

    // result[1] = N_{,u},  result[2] = N_{,v}
    result.push_back(basis_derivs[1 * S + 0]);
    result.push_back(basis_derivs[0 * S + 1]);

    // Compute Jacobian (area element) at each quadrature point
    for (Index q = 0; q < Q; ++q)
    {
        const T g11 = a_1.row(q).squaredNorm();
        const T g12 = a_1.row(q).dot(a_2.row(q));
        const T g22 = a_2.row(q).squaredNorm();
        jacobian(q) = std::sqrt(g11 * g22 - g12 * g12);
    }

    if (order >= 2)
    {
        // Second geometry derivatives  a_{α,β} = ∂²x/∂ξ^α∂ξ^β   (Q × 3)
        ColMatrix<T, 3> a_11 = basis_derivs[2 * S + 0] * act_pts;
        ColMatrix<T, 3> a_12 = basis_derivs[1 * S + 1] * act_pts;
        ColMatrix<T, 3> a_22 = basis_derivs[0 * S + 2] * act_pts;

        // Aliases for the parametric derivatives we will correct
        const Matrix<T>& N_u  = basis_derivs[1 * S + 0];
        const Matrix<T>& N_v  = basis_derivs[0 * S + 1];
        const Matrix<T>& N_uu = basis_derivs[2 * S + 0];
        const Matrix<T>& N_uv = basis_derivs[1 * S + 1];
        const Matrix<T>& N_vv = basis_derivs[0 * S + 2];

        // Allocate second-order output
        result.push_back(Matrix<T>(Q, K));  // result[3] = N_{;uu}
        result.push_back(Matrix<T>(Q, K));  // result[4] = N_{;uv}
        result.push_back(Matrix<T>(Q, K));  // result[5] = N_{;vv}

        for (Index q = 0; q < Q; ++q)
        {
            // Metric tensor
            const T g11 = a_1.row(q).squaredNorm();
            const T g12 = a_1.row(q).dot(a_2.row(q));
            const T g22 = a_2.row(q).squaredNorm();
            const T det_g = g11 * g22 - g12 * g12;
            const T inv_det = T(1) / det_g;

            // Contravariant (inverse) metric
            const T gi11 =  g22 * inv_det;
            const T gi12 = -g12 * inv_det;
            const T gi22 =  g11 * inv_det;

            // Christoffel symbols of the first kind:
            //   Γ_{αβ,δ} = a_{α,β} · a_δ
            const T C_111 = a_11.row(q).dot(a_1.row(q));
            const T C_112 = a_11.row(q).dot(a_2.row(q));
            const T C_121 = a_12.row(q).dot(a_1.row(q));
            const T C_122 = a_12.row(q).dot(a_2.row(q));
            const T C_221 = a_22.row(q).dot(a_1.row(q));
            const T C_222 = a_22.row(q).dot(a_2.row(q));

            // Christoffel symbols of the second kind:
            //   Γ^γ_{αβ} = g^{γδ} Γ_{αβ,δ}
            const T G1_11 = gi11 * C_111 + gi12 * C_112;
            const T G2_11 = gi12 * C_111 + gi22 * C_112;

            const T G1_12 = gi11 * C_121 + gi12 * C_122;
            const T G2_12 = gi12 * C_121 + gi22 * C_122;

            const T G1_22 = gi11 * C_221 + gi12 * C_222;
            const T G2_22 = gi12 * C_221 + gi22 * C_222;

            // Covariant second derivatives:
            //   N_{;αβ} = N_{,αβ} − Γ^1_{αβ} N_{,u} − Γ^2_{αβ} N_{,v}
            result[3].row(q) = N_uu.row(q)
                             - G1_11 * N_u.row(q) - G2_11 * N_v.row(q);
            result[4].row(q) = N_uv.row(q)
                             - G1_12 * N_u.row(q) - G2_12 * N_v.row(q);
            result[5].row(q) = N_vv.row(q)
                             - G1_22 * N_u.row(q) - G2_22 * N_v.row(q);
        }
    }

    return {std::move(result), std::move(jacobian)};
}

template <std::floating_point T>
ColMatrix<T, 3> SurfacePatch<T>::eval_geometry(const ColMatrix<T, 2>& points,
                                               Index span) const
{
    auto spans = decode_span(span);
    auto basis_fns = tensor_product_.eval_on_span(points, spans, 0);
    auto act_pts = this->active_control_pts(spans);
    return basis_fns[0] * act_pts;  // (Q × K) * (K × 3)
}

template <std::floating_point T>
ColMatrix<T, 3> SurfacePatch<T>::eval_physical_points(
    const QuadratureRule<T, 2>& quadrature) const
{
    auto intervals = tensor_product_.num_intervals();
    const Index total_elements = intervals[0] * intervals[1];
    const Index Q = static_cast<Index>(quadrature.num_points());

    ColMatrix<T, 3> result(total_elements * Q, 3);
    Index out = 0;

    for (Index elem_idx = 0; elem_idx < total_elements; ++elem_idx)
    {
        auto spans = decode_span(elem_idx);

        // Check for zero-volume elements
        std::array<T, 2> lo, hi;
        bool zero_volume = false;
        for (std::size_t i = 0; i < 2; ++i)
        {
            auto [l, h] = this->basis(i).knot_vector().span_bounds(spans[i]);
            lo[i] = l;
            hi[i] = h;
            if (std::abs(h - l) < T(1e-14))
            {
                zero_volume = true;
                break;
            }
        }
        if (zero_volume) continue;

        auto [mapped_pts, mapped_weights] = quadrature.map_to_domain(lo, hi);
        auto geom = this->eval_geometry(mapped_pts, elem_idx);
        result.block(out, 0, Q, 3) = geom;
        out += Q;
    }

    result.conservativeResize(out, Eigen::NoChange);
    return result;
}

template <std::floating_point T>
static std::vector<T> greville(Ptr<const Basis<T>> bs)
{
    const Index n = bs->num_basis();
    const Index p = bs->degree();

    auto bspline = std::dynamic_pointer_cast<const BSpline<T>>(bs);
    if (!bspline) {
        throw std::runtime_error(
            "Greville abscissae requires a BSpline basis implementation.");
    }

    const auto& knots_vec = bspline->knots();
    std::vector<T> xi(n);
    for (Index i = 0; i < n; ++i) {
        T sum = T(0);
        for (Index j = 1; j <= p; ++j)
            sum += knots_vec[i + j];
        xi[i] = sum / static_cast<T>(p);
    }
    return xi;
}

template <std::floating_point T>
SurfacePatch<T> rectangle(Ptr<const Basis<T>> basis_u,
                           Ptr<const Basis<T>> basis_v,
                           T width,
                           T height)
{
    auto xi_u = greville(basis_u);
    auto xi_v = greville(basis_v);

    const Index nu = basis_u->num_basis();
    const Index nv = basis_v->num_basis();

    // Normalise abscissae to [0, 1]
    T u_min = xi_u.front(), u_max = xi_u.back();
    T v_min = xi_v.front(), v_max = xi_v.back();
    T u_range = (std::abs(u_max - u_min) > T(1e-14)) ? (u_max - u_min) : T(1);
    T v_range = (std::abs(v_max - v_min) > T(1e-14)) ? (v_max - v_min) : T(1);

    // u-fastest ordering:  global = i_u + i_v * nu
    ColMatrix<T, 3> P = ColMatrix<T, 3>::Zero(nu * nv, 3);
    for (Index iv = 0; iv < nv; ++iv) {
        T y = height * (xi_v[iv] - v_min) / v_range;
        for (Index iu = 0; iu < nu; ++iu) {
            T x = width * (xi_u[iu] - u_min) / u_range;
            P.row(iu + iv * nu) << x, y, T(0);
        }
    }

    return SurfacePatch<T>(basis_u, basis_v, P);
}

// === Template Instantiations ========================================================

template class SurfacePatch<double>;
template SurfacePatch<double> rectangle<double>(Ptr<const Basis<double>>,
                                                 Ptr<const Basis<double>>,
                                                 double, double);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class SurfacePatch<float>;
template SurfacePatch<float> rectangle<float>(Ptr<const Basis<float>>,
                                               Ptr<const Basis<float>>,
                                               float, float);
#endif

} // namespace pyck
