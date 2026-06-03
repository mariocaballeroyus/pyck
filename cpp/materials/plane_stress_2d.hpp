#ifndef PYCK_PLANE_STRESS_2D_HPP
#define PYCK_PLANE_STRESS_2D_HPP

#include "material.hpp"

namespace pyck
{

/**
 * @brief Isotropic linear-elastic plane-stress material for plates and shells.
 *
 *        The material stiffness (constitutive) tensor is given by:
 *        C^{αβγδ} = (E/(1−ν²)) [ ½(1−ν)(A^{αγ}A^{βδ} + A^{αδ}A^{βγ})
 *                              + ν A^{αβ} A^{γδ} ]
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
        : E_(E), nu_(nu), t_(t), k_(k), rho_(rho),
          mu_(E / (T(2) * (T(1) + nu))),
          lambda_tilde_(E * nu / (T(1) - nu * nu)),
          lp2m_(lambda_tilde_ + T(2) * mu_),
          bending_scale_(t * t * t / T(12)),
          shear_scale_(k * mu_ * t)
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
    T shear_modulus()            const override { return mu_; }
    T thickness()                const          { return t_; }
    T shear_correction_factor()  const          { return k_; }
    T density()                  const override { return rho_; }

    T bending_stiffness() const override { return lp2m_ * bending_scale_; }
    T shear_stiffness()   const override { return shear_scale_; }

    std::string name() const override { return "PlaneStress2d"; }

    /**
     * @brief Plane-stress constitutive matrix (Cartesian, flat). Reference
     *        flat-patch C-matrix; curved-surface assembly uses elasticity_voigt.
     */
    Matrix<T> constitutive_matrix() const
    {
        Matrix<T> C = Matrix<T>::Zero(3, 3);
        C(0, 0) = lp2m_;
        C(0, 1) = lambda_tilde_;
        C(1, 0) = lambda_tilde_;
        C(1, 1) = lp2m_;
        C(2, 2) = mu_;
        return C;
    }

    /**
     * @brief 3×3 Voigt form of the surface elasticity (stiffness) tensor
     *        C^{αβγδ} = λ̃ A^{αβ} A^{γδ} + μ (A^{αγ} A^{βδ} + A^{αδ} A^{βγ}),
     *        with plane-stress Lamé coefficients μ = E/(2(1+ν)) and
     *        λ̃ = Eν/(1-ν²).
     */
    Eigen::Matrix<T, 3, 3> elasticity_voigt(const StaticVector<T, 3>& metric_inv) const
    {
        const T g11 = metric_inv(0);
        const T g12 = metric_inv(1);
        const T g22 = metric_inv(2);

        Eigen::Matrix<T, 3, 3> D;
        D(0, 0) = lp2m_ * g11 * g11;
        D(1, 1) = lp2m_ * g22 * g22;
        D(0, 1) = lambda_tilde_ * g11 * g22 + T(2) * mu_ * g12 * g12;
        D(0, 2) = lp2m_ * g11 * g12;
        D(1, 2) = lp2m_ * g22 * g12;
        D(2, 2) = (lambda_tilde_ + mu_) * g12 * g12 + mu_ * g11 * g22;
        D(1, 0) = D(0, 1);
        D(2, 0) = D(0, 2);
        D(2, 1) = D(1, 2);
        return D;
    }

    /// Membrane Voigt 3×3 D_m = t · C at one quadrature point.
    Eigen::Matrix<T, 3, 3> membrane_voigt(const StaticVector<T, 3>& metric_inv) const
    {
        return t_ * elasticity_voigt(metric_inv);
    }

    /// Bending Voigt 3×3 D_b = t³/12 · C at one quadrature point.
    Eigen::Matrix<T, 3, 3> bending_voigt(const StaticVector<T, 3>& metric_inv) const
    {
        return bending_scale_ * elasticity_voigt(metric_inv);
    }

    /// Transverse-shear Voigt 2×2 D_s = κ_s G t · A^{αβ} at one quadrature point.
    Eigen::Matrix<T, 2, 2> shear_voigt(const StaticVector<T, 3>& metric_inv) const
    {
        Eigen::Matrix<T, 2, 2> Ds;
        Ds(0, 0) = shear_scale_ * metric_inv(0);
        Ds(0, 1) = shear_scale_ * metric_inv(1);
        Ds(1, 0) = Ds(0, 1);
        Ds(1, 1) = shear_scale_ * metric_inv(2);
        return Ds;
    }

private:
    T E_;
    T nu_;
    T t_;
    T k_;
    T rho_;
    T mu_;             ///< Plane-stress shear modulus G = E/(2(1+ν)).
    T lambda_tilde_;   ///< Plane-stress Lamé λ̃ = Eν/(1-ν²).
    T lp2m_;           ///< λ̃ + 2μ = E/(1-ν²).
    T bending_scale_;  ///< t³/12 — bending-stiffness thickness weight.
    T shear_scale_;    ///< κ_s · μ · t — transverse-shear stiffness scale.
};

} // namespace pyck

#endif // PYCK_PLANE_STRESS_2D_HPP
