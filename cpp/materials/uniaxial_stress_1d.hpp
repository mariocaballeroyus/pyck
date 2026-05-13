#ifndef PYCK_UNIAXIAL_STRESS_1D_HPP
#define PYCK_UNIAXIAL_STRESS_1D_HPP

#include "material.hpp"

namespace pyck
{

/**
 * @brief Isotropic linear-elastic uniaxial-stress material for beams and 3D
 *        frames.
 *
 * Section axes: 1 = beam axis, 2 = strong (primary) bending axis,
 *               3 = weak (secondary) bending axis (for 3D frames only).
 *
 * Required: E, ν, A, I22. The planar-bending element families
 * (Euler-Bernoulli, Timoshenko 1p/2p) use only these.
 *
 * Optional: k22 (shear correction, default 5/6), ρ (density, default 0),
 *           I33, k33, J (default 0). A 3D-frame element should validate
 *           that I33, k33, J are non-zero at construction time.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class UniaxialStress1d : public Material<T, 1>
{
public:
    /**
     * @brief Construct a UniaxialStress1d.
     *
     * @param E    Young's modulus.                          (required, > 0)
     * @param nu   Poisson's ratio.                          (required, in (-1, 0.5))
     * @param A    Cross-sectional area.                     (required, > 0)
     * @param I22  Second moment of area about axis 2.       (required, > 0)
     * @param k22  Shear correction factor along axis 2.     (optional, default 5/6, > 0)
     * @param rho  Mass density.                             (optional, default 0, ≥ 0)
     * @param I33  Second moment of area about axis 3.       (optional, default 0, ≥ 0)
     * @param k33  Shear correction factor along axis 3.     (optional, default 0, ≥ 0)
     * @param J    Torsion constant (St-Venant).             (optional, default 0, ≥ 0)
     */
    UniaxialStress1d(T E, T nu, T A, T I22,
                     T k22 = T(5.0 / 6.0),
                     T rho = T(0),
                     T I33 = T(0),
                     T k33 = T(0),
                     T J   = T(0))
        : E_(E), nu_(nu), A_(A),
          I22_(I22), I33_(I33),
          k22_(k22), k33_(k33),
          J_(J), rho_(rho)
    {
        if (E_   <= 0)                       throw std::invalid_argument("UniaxialStress1d: E must be positive.");
        if (nu_  <= T(-1) || nu_ >= T(0.5))  throw std::invalid_argument("UniaxialStress1d: nu must be in (-1, 0.5).");
        if (A_   <= 0)                       throw std::invalid_argument("UniaxialStress1d: A must be positive.");
        if (I22_ <= 0)                       throw std::invalid_argument("UniaxialStress1d: I22 must be positive.");
        if (k22_ <= 0)                       throw std::invalid_argument("UniaxialStress1d: k22 must be positive.");
        if (rho_ < 0)                        throw std::invalid_argument("UniaxialStress1d: rho must be non-negative.");
        if (I33_ < 0)                        throw std::invalid_argument("UniaxialStress1d: I33 must be non-negative.");
        if (k33_ < 0)                        throw std::invalid_argument("UniaxialStress1d: k33 must be non-negative.");
        if (J_   < 0)                        throw std::invalid_argument("UniaxialStress1d: J must be non-negative.");

        G_ = E_ / (T(2) * (T(1) + nu_));
    }

    T youngs_modulus() const override { return E_; }
    T shear_modulus()  const override { return G_; }
    T poisson_ratio()  const override { return nu_; }
    T density()        const override { return rho_; }

    T section_area()         const { return A_; }
    T moment_inertia_22()    const { return I22_; }
    T moment_inertia_33()    const { return I33_; }
    T shear_coefficient_22() const { return k22_; }
    T shear_coefficient_33() const { return k33_; }
    T torsion_constant()     const { return J_; }

    /// EI about the primary bending axis (axis 2). 3D-frame elements should
    /// use moment_inertia_22() / _33() and torsion_constant() directly.
    T bending_stiffness() const override { return E_ * I22_; }

    /// κ G A along the primary shear axis (axis 2).
    T shear_stiffness() const override { return k22_ * G_ * A_; }

    std::string name() const override { return "UniaxialStress1d"; }

private:
    T E_;
    T nu_;
    T G_;
    T A_;
    T I22_, I33_;
    T k22_, k33_;
    T J_;
    T rho_;
};

} // namespace pyck

#endif // PYCK_UNIAXIAL_STRESS_1D_HPP
