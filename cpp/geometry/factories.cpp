#include "factories.hpp"
#include "../basis/bspline.hpp"
#include <cmath>

namespace pyck
{

/**
 * @brief Helper to compute Greville abscissae for any basis.
 *
 * Delegates to the basis's virtual greville_abscissae() method, which works
 * for both B-splines and NURBS (the Greville points depend only on the knot
 * vector and degree).
 *
 * @param bs The basis for which to compute the Greville abscissae.
 * @return A vector of Greville abscissae.
 */
template <std::floating_point T>
std::vector<T> greville_abscissae(Ptr<const Basis<T>> bs)
{
    Vector<T> g = bs->greville_abscissae();
    return std::vector<T>(g.data(), g.data() + g.size());
}

template <std::floating_point T>
Patch<T, 1> line_segment(Ptr<const pyck::Basis<T>> basis, T length)
{
    std::vector<T> xi = greville_abscissae(basis);
    Index n = basis->num_basis();

    ColMatrix<T, 3> P = ColMatrix<T, 3>::Zero(n, 3);

    T u_min = xi.front();
    T u_max = xi.back();
    T u_range = u_max - u_min;

    if (u_range < 1e-14) {
        u_range = 1.0; 
    }

    for (Index i = 0; i < n; ++i) {
        T normalized_u = (xi[i] - u_min) / u_range;
        P(i, 0) = length * normalized_u;
    }

    return Patch<T, 1>(basis, P);
}

template <std::floating_point T>
Patch<T, 2> rectangle(Ptr<const Basis<T>> basis_u,
                           Ptr<const Basis<T>> basis_v,
                           T width,
                           T height)
{
    auto xi_u = greville_abscissae(basis_u);
    auto xi_v = greville_abscissae(basis_v);

    const Index nu = basis_u->num_basis();
    const Index nv = basis_v->num_basis();

    T u_min = xi_u.front(), u_max = xi_u.back();
    T v_min = xi_v.front(), v_max = xi_v.back();
    T u_range = (std::abs(u_max - u_min) > T(1e-14)) ? (u_max - u_min) : T(1);
    T v_range = (std::abs(v_max - v_min) > T(1e-14)) ? (v_max - v_min) : T(1);

    ColMatrix<T, 3> P = ColMatrix<T, 3>::Zero(nu * nv, 3);
    for (Index iv = 0; iv < nv; ++iv) {
        T y = height * (xi_v[iv] - v_min) / v_range;
        for (Index iu = 0; iu < nu; ++iu) {
            T x = width * (xi_u[iu] - u_min) / u_range;
            P.row(iu + iv * nu) << x, y, T(0);
        }
    }

    return Patch<T, 2>(basis_u, basis_v, P);
}

// === Template Instantiations ========================================================

template Patch<double, 1> line_segment<double>(Ptr<const Basis<double>>, double);
template Patch<double, 2> rectangle<double>(Ptr<const Basis<double>>,
                                                 Ptr<const Basis<double>>,
                                                 double, double);

#ifdef PYCK_BUILD_SINGLE_PRECISION
template Patch<float, 1> line_segment<float>(Ptr<const Basis<float>>, float);
template Patch<float, 2> rectangle<float>(Ptr<const Basis<float>>,
                                               Ptr<const Basis<float>>,
                                               float, float);
#endif

} // namespace pyck
