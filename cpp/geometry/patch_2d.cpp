#include "patch.hpp"
#include "bspline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../quadrature/quadrature.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
Patch<T, d>::Patch(Ptr<const Basis<T>> basis_u, Ptr<const Basis<T>> basis_v, const ColMatrix<T, 3>& control_pts) requires(d == 2)
    : control_pts_(control_pts),
      tensor_product_(std::move(basis_u), std::move(basis_v)),
      dof_mapper_({tensor_product_.basis(0).num_basis(), tensor_product_.basis(1).num_basis()},
                  {tensor_product_.basis(0).degree(),    tensor_product_.basis(1).degree()})
{
    if (control_pts.cols() != 3) {
        throw std::invalid_argument("Patch<T, 2>: Control points must be embedded in 3D space.");
    }
    const Index expected_n = this->tensor_product_.basis(0).num_basis() * this->tensor_product_.basis(1).num_basis();
    const Index actual_n = static_cast<Index>(control_pts.rows());
    if (actual_n != expected_n) {
        throw std::invalid_argument("Patch<T, 2>: Dimension mismatch.");
    }

    // Initialize Greville points and spans
    Vector<T> gu = this->tensor_product_.basis(0).greville_abscissae();
    Vector<T> gv = this->tensor_product_.basis(1).greville_abscissae();
    Index nu = gu.size();
    Index nv = gv.size();
    
    greville_points_.resize(nu * nv, 2);
    greville_spans_.resize(nu * nv);
    
    auto intervals = this->tensor_product_.num_intervals();
    for (Index i = 0; i < nu; ++i) {
        Index su = this->tensor_product_.basis(0).find_span(gu[i]);
        for (Index j = 0; j < nv; ++j) {
            Index flat = i * nv + j;
            greville_points_(flat, 0) = gu[i];
            greville_points_(flat, 1) = gv[j];
            Index sv = this->tensor_product_.basis(1).find_span(gv[j]);
            greville_spans_[flat] = su * intervals[1] + sv;
        }
    }
}

template <std::floating_point T, std::size_t d>
std::pair<std::vector<Matrix<T>>, Vector<T>> 
Patch<T, d>::eval_shape_functions_at_greville(Index dof_index, std::size_t order_in) const 
{
    ColMatrix<T, d> pt = greville_points_.row(dof_index);
    Index span = greville_spans_[dof_index];
    return eval_shape_functions(pt, span, order_in);
}

template <std::floating_point T, std::size_t d>
std::pair<std::vector<Matrix<T>>, Vector<T>> 
Patch<T, d>::eval_shape_functions(const ColMatrix<T, d>& points, Index span, std::size_t order_in) const requires(d == 2)
{
    auto spans = this->decode_span(span);
    Index order = std::max(Index(1), std::min(static_cast<Index>(order_in), Index(3)));
    const Index Q = points.rows();

    // Evaluate parametric basis derivatives up to 'order'.
    // For order=2 the tensor product returns (order+1)^2 = 9 matrices.
    // Layout:  idx = du * (order+1) + dv
    //   (0,0)=N   (0,1)=N_v   (0,2)=N_vv
    //   (1,0)=N_u (1,1)=N_uv  (1,2)=N_uvv
    //   (2,0)=N_uu(2,1)=N_uuv (2,2)=N_uuvv
    auto basis_derivs = this->tensor_product().eval_on_span(points, spans, order);
    const Index K = basis_derivs[0].cols();
    const Index S = order + 1;  // stride for flat index

    auto act_pts = this->active_control_pts(spans);

    ColMatrix<T, 3> a_1 = basis_derivs[1 * S + 0] * act_pts;  // N_{,u} * P
    ColMatrix<T, 3> a_2 = basis_derivs[0 * S + 1] * act_pts;  // N_{,v} * P

    const size_t num_results = (order == 1) ? 3 : (order == 2 ? 6 : 10);
    std::vector<Matrix<T>> result(num_results);
    Vector<T> jacobian(Q);

    result[0] = basis_derivs[0];

    // result[1] = N_{,u},  result[2] = N_{,v}
    result[1] = basis_derivs[1 * S + 0];
    result[2] = basis_derivs[0 * S + 1];

    // Compute Jacobian (area element) at each quadrature point:
    // J = ||a_1 × a_2|| = sqrt(g11 * g22 - g12^2)
    for (Index q = 0; q < Q; ++q)
    {
        const T g11 = a_1.row(q).squaredNorm();
        const T g12 = a_1.row(q).dot(a_2.row(q));
        const T g22 = a_2.row(q).squaredNorm();
        jacobian(q) = std::sqrt(g11 * g22 - g12 * g12);

        if (order >= 1) {
            result[1].row(q) /= std::sqrt(g11);
            result[2].row(q) /= std::sqrt(g22);
        }
    }

    if (order >= 2)
    {
        // Second geometry derivatives  a_{α,β} = ∂²x/∂ξ^α∂ξ^β   (Q × 3)
        ColMatrix<T, 3> a_11 = basis_derivs[2 * S + 0] * act_pts;
        ColMatrix<T, 3> a_12 = basis_derivs[1 * S + 1] * act_pts;
        ColMatrix<T, 3> a_22 = basis_derivs[0 * S + 2] * act_pts;

        const Matrix<T>& N_u  = basis_derivs[1 * S + 0];
        const Matrix<T>& N_v  = basis_derivs[0 * S + 1];
        const Matrix<T>& N_uu = basis_derivs[2 * S + 0];
        const Matrix<T>& N_uv = basis_derivs[1 * S + 1];
        const Matrix<T>& N_vv = basis_derivs[0 * S + 2];

        result[3] = Matrix<T>(Q, K);  // result[3] = N_{;uu}
        result[4] = Matrix<T>(Q, K);  // result[4] = N_{;uv}
        result[5] = Matrix<T>(Q, K);  // result[5] = N_{;vv}

        for (Index q = 0; q < Q; ++q)
        {
            // Metric tensor components: g_{αβ} = a_α · a_β
            const T g11 = a_1.row(q).squaredNorm();
            const T g12 = a_1.row(q).dot(a_2.row(q));
            const T g22 = a_2.row(q).squaredNorm();
            const T det_g = g11 * g22 - g12 * g12;
            const T inv_det = T(1) / det_g;

            // Contravariant (inverse) metric: g^{αβ}
            const T gi11 =  g22 * inv_det;
            const T gi12 = -g12 * inv_det;
            const T gi22 =  g11 * inv_det;

            // Christoffel symbols of the first kind: Γ_{αβ,δ} = a_{α,β} · a_δ
            const T C_111 = a_11.row(q).dot(a_1.row(q));
            const T C_112 = a_11.row(q).dot(a_2.row(q));
            const T C_121 = a_12.row(q).dot(a_1.row(q));
            const T C_122 = a_12.row(q).dot(a_2.row(q));
            const T C_221 = a_22.row(q).dot(a_1.row(q));
            const T C_222 = a_22.row(q).dot(a_2.row(q));

            // Christoffel symbols of the second kind: Γ^γ_{αβ} = g^{γδ} Γ_{αβ,δ}
            const T G1_11 = gi11 * C_111 + gi12 * C_112;
            const T G2_11 = gi12 * C_111 + gi22 * C_112;

            const T G1_12 = gi11 * C_121 + gi12 * C_122;
            const T G2_12 = gi12 * C_121 + gi22 * C_122;

            const T G1_22 = gi11 * C_221 + gi12 * C_222;
            const T G2_22 = gi12 * C_221 + gi22 * C_222;

            // Second covariant derivatives:
            //   N_{;αβ} = N_{,αβ} − Γ^γ_{αβ} N_{,γ}
            // result[3] = N_{;uu},  result[4] = N_{;uv},  result[5] = N_{;vv}
            result[3].row(q) = (N_uu.row(q) - G1_11 * N_u.row(q) - G2_11 * N_v.row(q)) / g11;
            result[4].row(q) = (N_uv.row(q) - G1_12 * N_u.row(q) - G2_12 * N_v.row(q)) / std::sqrt(g11 * g22);
            result[5].row(q) = (N_vv.row(q) - G1_22 * N_u.row(q) - G2_22 * N_v.row(q)) / g22;
        }
    }

    if (order >= 3)
    {
        ColMatrix<T, 3> a_111 = basis_derivs[3 * S + 0] * act_pts;
        ColMatrix<T, 3> a_112 = basis_derivs[2 * S + 1] * act_pts;
        ColMatrix<T, 3> a_122 = basis_derivs[1 * S + 2] * act_pts;
        ColMatrix<T, 3> a_222 = basis_derivs[0 * S + 3] * act_pts;
        ColMatrix<T, 3> a_11 = basis_derivs[2 * S + 0] * act_pts;
        ColMatrix<T, 3> a_12 = basis_derivs[1 * S + 1] * act_pts;
        ColMatrix<T, 3> a_22 = basis_derivs[0 * S + 2] * act_pts;

        const Matrix<T>& N_u   = basis_derivs[1 * S + 0];
        const Matrix<T>& N_v   = basis_derivs[0 * S + 1];
        const Matrix<T>& N_uu  = basis_derivs[2 * S + 0];
        const Matrix<T>& N_uv  = basis_derivs[1 * S + 1];
        const Matrix<T>& N_vv  = basis_derivs[0 * S + 2];
        const Matrix<T>& N_uuu = basis_derivs[3 * S + 0];
        const Matrix<T>& N_uuv = basis_derivs[2 * S + 1];
        const Matrix<T>& N_uvv = basis_derivs[1 * S + 2];
        const Matrix<T>& N_vvv = basis_derivs[0 * S + 3];

        result[6] = Matrix<T>(Q, K);  // result[6] = N_{;uuu}
        result[7] = Matrix<T>(Q, K);  // result[7] = N_{;uuv}
        result[8] = Matrix<T>(Q, K);  // result[8] = N_{;uvv}
        result[9] = Matrix<T>(Q, K);  // result[9] = N_{;vvv}

        for (Index q = 0; q < Q; ++q)
        {
            const T g11 = a_1.row(q).squaredNorm();
            const T g12 = a_1.row(q).dot(a_2.row(q));
            const T g22 = a_2.row(q).squaredNorm();
            const T det_g = g11 * g22 - g12 * g12;
            const T inv_det = T(1) / det_g;

            const T gi11 =  g22 * inv_det;
            const T gi12 = -g12 * inv_det;
            const T gi22 =  g11 * inv_det;

            const T C_111 = a_11.row(q).dot(a_1.row(q));
            const T C_112 = a_11.row(q).dot(a_2.row(q));
            const T C_121 = a_12.row(q).dot(a_1.row(q));
            const T C_122 = a_12.row(q).dot(a_2.row(q));
            const T C_221 = a_22.row(q).dot(a_1.row(q));
            const T C_222 = a_22.row(q).dot(a_2.row(q));

            const T G1_11 = gi11 * C_111 + gi12 * C_112;
            const T G2_11 = gi12 * C_111 + gi22 * C_112;
            const T G1_12 = gi11 * C_121 + gi12 * C_122;
            const T G2_12 = gi12 * C_121 + gi22 * C_122;
            const T G1_22 = gi11 * C_221 + gi12 * C_222;
            const T G2_22 = gi12 * C_221 + gi22 * C_222;

            // Derivatives of Christoffel symbols of the first kind: C_{αβγ,δ} = ∂C_{αβγ}/∂ξ^δ
            const T dC_111_1 = a_111.row(q).dot(a_1.row(q)) + a_11.row(q).dot(a_11.row(q));
            const T dC_112_1 = a_111.row(q).dot(a_2.row(q)) + a_11.row(q).dot(a_12.row(q));
            // dC_{111,2}:
            const T dC_111_2 = a_112.row(q).dot(a_1.row(q)) + a_11.row(q).dot(a_12.row(q));
            // dC_{112,2}:
            const T dC_112_2 = a_112.row(q).dot(a_2.row(q)) + a_11.row(q).dot(a_22.row(q));
            // dC_{121,1}:
            const T dC_121_1 = a_112.row(q).dot(a_1.row(q)) + a_12.row(q).dot(a_11.row(q));
            // dC_{122,1}:
            const T dC_122_1 = a_112.row(q).dot(a_2.row(q)) + a_12.row(q).dot(a_12.row(q));
            // dC_{121,2}:
            const T dC_121_2 = a_122.row(q).dot(a_1.row(q)) + a_12.row(q).dot(a_12.row(q));
            // dC_{122,2}:
            const T dC_122_2 = a_122.row(q).dot(a_2.row(q)) + a_12.row(q).dot(a_22.row(q));
            // dC_{221,1}:
            const T dC_221_1 = a_122.row(q).dot(a_1.row(q)) + a_22.row(q).dot(a_11.row(q));
            // dC_{222,1}:
            const T dC_222_1 = a_122.row(q).dot(a_2.row(q)) + a_22.row(q).dot(a_12.row(q));
            // dC_{221,2}:
            const T dC_221_2 = a_222.row(q).dot(a_1.row(q)) + a_22.row(q).dot(a_12.row(q));
            // dC_{222,2}:
            const T dC_222_2 = a_222.row(q).dot(a_2.row(q)) + a_22.row(q).dot(a_22.row(q));

            // Derivatives of the metric tensor: g_{αβ,δ}
            // g_{11,1}:
            const T dg11_1 = T(2) * a_11.row(q).dot(a_1.row(q));
            // g_{11,2}:
            const T dg11_2 = T(2) * a_12.row(q).dot(a_1.row(q));
            // g_{12,1}:
            const T dg12_1 = a_11.row(q).dot(a_2.row(q)) + a_1.row(q).dot(a_12.row(q));
            // g_{12,2}:
            const T dg12_2 = a_12.row(q).dot(a_2.row(q)) + a_1.row(q).dot(a_22.row(q));
            // g_{22,1}:
            const T dg22_1 = T(2) * a_12.row(q).dot(a_2.row(q));
            // g_{22,2}:
            const T dg22_2 = T(2) * a_22.row(q).dot(a_2.row(q));

            // Derivatives of the inverse metric: g^{αβ}_{,δ} = -g^{αμ} g_{μν,δ} g^{νβ}
            // g^{11}_{,1}:
            const T dgi11_1 = -(gi11 * dg11_1 * gi11 + gi11 * dg12_1 * gi12
                              + gi12 * dg12_1 * gi11 + gi12 * dg22_1 * gi12);
            // g^{12}_{,1}:
            const T dgi12_1 = -(gi11 * dg11_1 * gi12 + gi11 * dg12_1 * gi22
                              + gi12 * dg12_1 * gi12 + gi12 * dg22_1 * gi22);
            // g^{22}_{,1}:
            const T dgi22_1 = -(gi12 * dg11_1 * gi12 + gi12 * dg12_1 * gi22
                              + gi22 * dg12_1 * gi12 + gi22 * dg22_1 * gi22);
            // g^{11}_{,2}:
            const T dgi11_2 = -(gi11 * dg11_2 * gi11 + gi11 * dg12_2 * gi12
                              + gi12 * dg12_2 * gi11 + gi12 * dg22_2 * gi12);
            // g^{12}_{,2}:
            const T dgi12_2 = -(gi11 * dg11_2 * gi12 + gi11 * dg12_2 * gi22
                              + gi12 * dg12_2 * gi12 + gi12 * dg22_2 * gi22);
            // g^{22}_{,2}:
            const T dgi22_2 = -(gi12 * dg11_2 * gi12 + gi12 * dg12_2 * gi22
                              + gi22 * dg12_2 * gi12 + gi22 * dg22_2 * gi22);

            // Derivatives of Christoffel symbols of the second kind: Γ^γ_{αβ,δ} = ∂(g^{γλ} Γ_{αβ,λ})/∂ξ^δ
            // Γ^δ_{11,1}:
            const T dG1_11_1 = dgi11_1 * C_111 + dgi12_1 * C_112
                             + gi11 * dC_111_1 + gi12 * dC_112_1; // Γ^1_{11,1}
            const T dG2_11_1 = dgi12_1 * C_111 + dgi22_1 * C_112
                             + gi12 * dC_111_1 + gi22 * dC_112_1; // Γ^2_{11,1}

            // Γ^δ_{11,2}:
            const T dG1_11_2 = dgi11_2 * C_111 + dgi12_2 * C_112
                             + gi11 * dC_111_2 + gi12 * dC_112_2; // Γ^1_{11,2}
            const T dG2_11_2 = dgi12_2 * C_111 + dgi22_2 * C_112
                             + gi12 * dC_111_2 + gi22 * dC_112_2; // Γ^2_{11,2}

            // Γ^δ_{12,1}:
            const T dG1_12_1 = dgi11_1 * C_121 + dgi12_1 * C_122
                             + gi11 * dC_121_1 + gi12 * dC_122_1; // Γ^1_{12,1}
            const T dG2_12_1 = dgi12_1 * C_121 + dgi22_1 * C_122
                             + gi12 * dC_121_1 + gi22 * dC_122_1; // Γ^2_{12,1}

            // Γ^δ_{12,2}:
            const T dG1_12_2 = dgi11_2 * C_121 + dgi12_2 * C_122
                             + gi11 * dC_121_2 + gi12 * dC_122_2; // Γ^1_{12,2}
            const T dG2_12_2 = dgi12_2 * C_121 + dgi22_2 * C_122
                             + gi12 * dC_121_2 + gi22 * dC_122_2; // Γ^2_{12,2}

            // Γ^δ_{22,1}:
            const T dG1_22_1 = dgi11_1 * C_221 + dgi12_1 * C_222
                             + gi11 * dC_221_1 + gi12 * dC_222_1; // Γ^1_{22,1}
            const T dG2_22_1 = dgi12_1 * C_221 + dgi22_1 * C_222
                             + gi12 * dC_221_1 + gi22 * dC_222_1; // Γ^2_{22,1}

            // Γ^δ_{22,2}:
            const T dG1_22_2 = dgi11_2 * C_221 + dgi12_2 * C_222
                             + gi11 * dC_221_2 + gi12 * dC_222_2; // Γ^1_{22,2}
            const T dG2_22_2 = dgi12_2 * C_221 + dgi22_2 * C_222
                             + gi12 * dC_221_2 + gi22 * dC_222_2; // Γ^2_{22,2}

            // Covariant third derivatives:
            //   N_{;αβγ} = N_{,αβγ}
            //            − Γ^δ_{αβ,γ} N_{,δ}
            //            − Γ^δ_{αβ} N_{,δγ}
            //            − Γ^δ_{αγ} N_{;δβ}
            //            − Γ^δ_{βγ} N_{;αδ}
            
            // Pre-compute the 2nd covariant derivatives used below
            // N_{;uu} = result[3], N_{;uv} = result[4], N_{;vv} = result[5]
            // (already computed in the order-2 block)
            auto N_cov_uu = result[3].row(q);
            auto N_cov_uv = result[4].row(q);
            auto N_cov_vv = result[5].row(q);

            // Third covariant derivatives:
            //   N_{;αβδ} = N_{,αβδ} - Γ^γ_{αβ,δ} N_{,γ} - Γ^γ_{αβ} N_{,γδ} 
            //              - Γ^γ_{αδ} N_{;γβ} - Γ^γ_{βδ} N_{;αγ}
            result[6].row(q) = N_uuu.row(q)
                - dG1_11_1 * N_u.row(q) - dG2_11_1 * N_v.row(q)
                - G1_11 * N_uu.row(q) - G2_11 * N_uv.row(q)
                - T(2) * G1_11 * N_cov_uu - T(2) * G2_11 * N_cov_uv; // N_{;uuu}

            result[7].row(q) = N_uuv.row(q)
                - dG1_11_2 * N_u.row(q) - dG2_11_2 * N_v.row(q)
                - G1_11 * N_uv.row(q) - G2_11 * N_vv.row(q)
                - T(2) * G1_12 * N_cov_uu - T(2) * G2_12 * N_cov_uv; // N_{;uuv}

            result[8].row(q) = N_uvv.row(q)
                - dG1_12_2 * N_u.row(q) - dG2_12_2 * N_v.row(q)
                - G1_12 * N_uv.row(q) - G2_12 * N_vv.row(q)
                - G1_12 * N_cov_uv - G2_12 * N_cov_vv
                - G1_22 * N_cov_uu - G2_22 * N_cov_uv; // N_{;uvv}

            result[9].row(q) = N_vvv.row(q)
                - dG1_22_2 * N_u.row(q) - dG2_22_2 * N_v.row(q)
                - G1_22 * N_uv.row(q) - G2_22 * N_vv.row(q)
                - T(2) * G1_22 * N_cov_uv - T(2) * G2_22 * N_cov_vv; // N_{;vvv}

            const T h1 = std::sqrt(g11);
            const T h2 = std::sqrt(g22);
            result[6].row(q) /= h1 * g11;
            result[7].row(q) /= g11 * h2;
            result[8].row(q) /= h1 * g22;
            result[9].row(q) /= g22 * h2;
        }
    }

    return {std::move(result), std::move(jacobian)};
}


template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::eval_geometry(const ColMatrix<T, d>& points, Index span) const requires(d == 2)
{
    auto spans = this->decode_span(span);
    auto basis_fns = this->tensor_product().eval_on_span(points, spans, 0);
    auto act_pts = this->active_control_pts(spans);
    return basis_fns[0] * act_pts;
}

template <std::floating_point T, std::size_t d>
ColMatrix<T, 3> Patch<T, d>::eval_physical_points(const QuadratureRule<T, d>& quadrature) const requires(d == 2)
{
    auto intervals = this->tensor_product().num_intervals();
    const Index total_elements = intervals[0] * intervals[1];
    const Index Q = static_cast<Index>(quadrature.num_points());

    ColMatrix<T, 3> result(total_elements * Q, 3);
    Index out = 0;

    for (Index elem_idx = 0; elem_idx < total_elements; ++elem_idx)
    {
        auto spans = this->decode_span(elem_idx);
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

template class Patch<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class Patch<float, 2>;
#endif

} // namespace pyck
