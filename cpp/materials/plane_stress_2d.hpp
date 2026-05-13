#ifndef PYCK_PLANE_STRESS_2D_HPP
#define PYCK_PLANE_STRESS_2D_HPP

#include "material.hpp"

namespace pyck
{

/**
 * @brief Isotropic linear-elastic plane-stress material for plates and shells.
 *
 *        The material stiffness (constitutive) tensor is given by:
 *        C^{αβγδ} = (E/(1−ν²)) [ ½(1−ν)(g^{αγ}g^{βδ} + g^{αδ}g^{βγ})
 *                              + ν g^{αβ} g^{γδ} ]
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlaneStress2d : public Material<T, 2>
{
public:
    /**
     * @brief Construct a plane-stress material.
     * @param E    Young's modulus.        (required, > 0)
     * @param nu   Poisson's ratio.        (required, in (-1, 0.5))
     * @param t    Thickness.              (required, > 0)
     * @param k    Shear correction factor (optional, default 5/6, > 0).
     * @param rho  Mass density.           (optional, default 0, ≥ 0).
     */
    PlaneStress2d(T E, T nu, T t, T k = T(5.0 / 6.0), T rho = T(0))
        : E_(E), nu_(nu), t_(t), k_(k), rho_(rho)
    {
        if (E_ <= 0)
            throw std::invalid_argument("PlaneStress2d: E must be positive.");
        if (nu_ <= T(-1) || nu_ >= T(0.5))
            throw std::invalid_argument("PlaneStress2d: nu must be in (-1, 0.5).");
        if (t_ <= 0)
            throw std::invalid_argument("PlaneStress2d: t must be positive.");
        if (k_ <= 0)
            throw std::invalid_argument("PlaneStress2d: k must be positive.");
        if (rho_ < 0)
            throw std::invalid_argument("PlaneStress2d: rho must be non-negative.");
    }

    T youngs_modulus()           const override { return E_; }
    T poisson_ratio()            const override { return nu_; }
    T shear_modulus()            const override { return E_ / (T(2) * (T(1) + nu_)); }
    T thickness()                const          { return t_; }
    T shear_correction_factor()  const          { return k_; }
    T density()                  const override { return rho_; }

    T bending_stiffness() const override { return (E_ * t_ * t_ * t_) / (T(12) * (T(1) - nu_ * nu_)); }
    T shear_stiffness()   const override { return k_ * shear_modulus() * t_; }

    std::string name() const override { return "PlaneStress2d"; }

    /**
     * @brief Plane-stress constitutive matrix (Cartesian, flat). Reference
     *        flat-patch C-matrix; curved-surface assembly uses surface_C_voigt.
     */
    Matrix<T> constitutive_matrix() const
    {
        Matrix<T> C = Matrix<T>::Zero(3, 3);
        const T factor = E_ / (T(1) - nu_ * nu_);
        C(0, 0) = factor;
        C(0, 1) = factor * nu_;
        C(1, 0) = factor * nu_;
        C(1, 1) = factor;
        C(2, 2) = factor * (T(1) - nu_) / T(2);
        return C;
    }

    /**
     * @brief 3×3 Voigt form of the surface elasticity (stiffness) tensor
     *        C^{αβγδ}, unscaled.
     */
    Eigen::Matrix<T, 3, 3> surface_C_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        const T g11 = g_inv(0);
        const T g12 = g_inv(1);
        const T g22 = g_inv(2);
        const T factor = E_ / (T(1) - nu_ * nu_);
        const T half_1mnu = (T(1) - nu_) / T(2);

        auto C = [&](T gac, T gbd, T gad, T gbc, T gab, T gcd) -> T {
            return factor * (half_1mnu * (gac * gbd + gad * gbc) + nu_ * gab * gcd);
        };

        Eigen::Matrix<T, 3, 3> D;
        D(0, 0) = C(g11, g11, g11, g11, g11, g11);
        D(0, 1) = C(g12, g12, g12, g12, g11, g22);
        D(0, 2) = C(g11, g12, g12, g11, g11, g12);
        D(1, 0) = D(0, 1);
        D(1, 1) = C(g22, g22, g22, g22, g22, g22);
        D(1, 2) = C(g12, g22, g22, g12, g22, g12);
        D(2, 0) = D(0, 2);
        D(2, 1) = D(1, 2);
        D(2, 2) = C(g11, g22, g12, g12, g12, g12);
        return D;
    }

    /// Membrane Voigt 3×3 D_m = t · C at one quadrature point.
    Eigen::Matrix<T, 3, 3> membrane_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        return t_ * surface_C_voigt(g_inv);
    }

    /// Bending Voigt 3×3 D_b = t³/12 · C at one quadrature point.
    Eigen::Matrix<T, 3, 3> bending_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        return (t_ * t_ * t_ / T(12)) * surface_C_voigt(g_inv);
    }

    /// Transverse-shear Voigt 2×2 D_s = κ_s G t · g^{αβ} at one quadrature point.
    Eigen::Matrix<T, 2, 2> shear_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        const T scale = k_ * shear_modulus() * t_;
        Eigen::Matrix<T, 2, 2> Ds;
        Ds(0, 0) = scale * g_inv(0);
        Ds(0, 1) = scale * g_inv(1);
        Ds(1, 0) = scale * g_inv(1);
        Ds(1, 1) = scale * g_inv(2);
        return Ds;
    }

private:
    T E_;
    T nu_;
    T t_;
    T k_;
    T rho_;
};

} // namespace pyck

#endif // PYCK_PLANE_STRESS_2D_HPP
