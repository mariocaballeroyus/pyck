#ifndef PYCK_PLANE_STRESS_SHELL_HPP
#define PYCK_PLANE_STRESS_SHELL_HPP

#include <Eigen/Dense>
#include "material.hpp"

namespace pyck
{

/**
 * @brief Isotropic linear-elastic plane-stress material for shell elements.
 *
 * Provides the membrane, bending and transverse-shear constitutive Voigt
 * matrices on a curved surface. Each matrix is built from the surface
 * 4-tensor
 *   H^{αβγδ} = (E/(1−ν²)) [ ½(1−ν)(g^{αγ}g^{βδ} + g^{αδ}g^{βγ})
 *                          + ν g^{αβ} g^{γδ} ]
 * contracted with the inverse metric g^{αβ} supplied by the caller — so
 * the matrices vary across the patch on curved surfaces.
 *
 * Voigt strain conventions (engineering shear):
 *   ε_voigt = [ ε_{11},  ε_{22},  2ε_{12} ]^T
 *   χ_voigt = [ χ_{11},  χ_{22},  2χ_{12} ]^T
 *   γ_voigt = [ γ_1,     γ_2 ]^T   (covariant components)
 * Stress/moment/shear-resultant Voigt are the dual-paired tensor
 * components, so that strain energy density is
 *   ½ ε^T D_m ε  +  ½ χ^T D_b χ  +  ½ γ^T D_s γ.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlaneStressShell : public Material<T, 2>
{
public:
    /**
     * @brief Construct a plane-stress shell material.
     * @param E   Young's modulus.
     * @param nu  Poisson's ratio.
     * @param t   Shell thickness.
     * @param k_s Transverse-shear correction factor (5/6 for rectangular cross-section).
     */
    PlaneStressShell(T E, T nu = T(0.3), T t = T(1.0), T k_s = T(5.0 / 6.0))
        : E_(E), nu_(nu), t_(t), k_s_(k_s)
    {
        if (E_  <= 0)              throw std::invalid_argument("PlaneStressShell: E must be positive.");
        if (nu_ <= -1.0 || nu_ >= 0.5) throw std::invalid_argument("PlaneStressShell: nu must be in (-1, 0.5).");
        if (t_  <= 0)              throw std::invalid_argument("PlaneStressShell: t must be positive.");
        if (k_s_ <= 0)             throw std::invalid_argument("PlaneStressShell: k_s must be positive.");
    }

    T youngs_modulus()         const override { return E_; }
    T poisson_ratio()          const override { return nu_; }
    T shear_modulus()          const override { return E_ / (T(2) * (T(1) + nu_)); }
    T thickness()              const          { return t_; }
    T shear_correction_factor()const          { return k_s_; }

    /// Membrane (Et) bending stiffness scalar — provided for symmetry with PlaneStress2d.
    T bending_stiffness() const override { return (E_ * t_ * t_ * t_) / (T(12) * (T(1) - nu_ * nu_)); }
    T shear_stiffness()   const override { return k_s_ * shear_modulus() * t_; }

    std::string name() const override { return "PlaneStressShell"; }

    /// Membrane Voigt 3×3 D_m at one quadrature point. Scaled by t.
    Matrix<T> membrane_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        return t_ * surface_H_voigt(g_inv);
    }

    /// Bending Voigt 3×3 D_b at one quadrature point. Scaled by t³/12.
    Matrix<T> bending_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        return (t_ * t_ * t_ / T(12)) * surface_H_voigt(g_inv);
    }

    /// Transverse-shear Voigt 2×2 D_s at one quadrature point: κ_s G t · g^{αβ}.
    Matrix<T> shear_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        const T scale = k_s_ * shear_modulus() * t_;
        Matrix<T> Ds(2, 2);
        Ds(0, 0) = scale * g_inv(0);   // g^{11}
        Ds(0, 1) = scale * g_inv(1);   // g^{12}
        Ds(1, 0) = scale * g_inv(1);
        Ds(1, 1) = scale * g_inv(2);   // g^{22}
        return Ds;
    }

private:
    /**
     * @brief Build the 3×3 Voigt form of the surface elasticity tensor
     *        H^{αβγδ} = (E/(1−ν²)) [½(1−ν)(g^{αγ}g^{βδ} + g^{αδ}g^{βγ})
     *                              + ν g^{αβ} g^{γδ}], unscaled.
     *
     * Voigt convention: rows / cols indexed (αβ) ∈ {(11), (22), (12)} with
     * engineering shear (the 2 in 2ε_{12} is absorbed into the strain
     * vector, NOT into the matrix). The matrix is symmetric.
     */
    Matrix<T> surface_H_voigt(const Eigen::Matrix<T, 3, 1>& g_inv) const
    {
        const T g11 = g_inv(0);
        const T g12 = g_inv(1);
        const T g22 = g_inv(2);
        const T factor = E_ / (T(1) - nu_ * nu_);
        const T half_1mnu = (T(1) - nu_) / T(2);

        // Helper for an arbitrary entry H^{αβγδ}.
        auto H = [&](T gac, T gbd, T gad, T gbc, T gab, T gcd) -> T {
            return factor * (half_1mnu * (gac * gbd + gad * gbc) + nu_ * gab * gcd);
        };

        Matrix<T> D(3, 3);
        // Row (αβ) = (1,1)
        D(0, 0) = H(g11, g11, g11, g11, g11, g11);                       // (11,11)
        D(0, 1) = H(g12, g12, g12, g12, g11, g22);                       // (11,22)
        D(0, 2) = H(g11, g12, g12, g11, g11, g12);                       // (11,12)
        // Row (αβ) = (2,2)
        D(1, 0) = D(0, 1);
        D(1, 1) = H(g22, g22, g22, g22, g22, g22);                       // (22,22)
        D(1, 2) = H(g12, g22, g22, g12, g22, g12);                       // (22,12)
        // Row (αβ) = (1,2)
        D(2, 0) = D(0, 2);
        D(2, 1) = D(1, 2);
        D(2, 2) = H(g11, g22, g12, g12, g12, g12);                       // (12,12)
        return D;
    }

    T E_;
    T nu_;
    T t_;
    T k_s_;
};

} // namespace pyck

#endif // PYCK_PLANE_STRESS_SHELL_HPP
