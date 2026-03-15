#ifndef PYCK_SLENDER_BEAM_1D_HPP
#define PYCK_SLENDER_BEAM_1D_HPP

#include "material.hpp"

namespace pyck
{

/**
 * @brief Material and section properties for 1D beam elements.
 *
 * This class encapsulates Young's modulus (E), shear modulus (G),
 * cross-sectional area (A), second moment of area (I),
 * and the shear correction factor (k).
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class SlenderBeam1d : public Material<T, 1>
{
public:
    /**
     * @brief Construct a `SlenderBeam1d`.
     *
     * @param E Young's modulus.
     * @param nu Poisson's ratio.
     * @param A Cross-sectional area.
     * @param I Second moment of area (moment of inertia).
     * @param k Shear correction factor (optional, defaults to 5/6).
     */
    SlenderBeam1d(T E, T nu, T A, T I, T k = T(5.0 / 6.0))
        : E_(E), nu_(nu), A_(A), I_(I), k_(k)
    {
        if (E_ <= 0) throw std::invalid_argument("SlenderBeam1d: E must be positive.");
        if (nu_ < -1 || nu_ >= 0.5) throw std::invalid_argument("SlenderBeam1d: nu must be in [-1, 0.5).");
        if (A_ <= 0) throw std::invalid_argument("SlenderBeam1d: A must be positive.");
        if (I_ <= 0) throw std::invalid_argument("SlenderBeam1d: I must be positive.");
        
        G_ = E_ / (T(2) * (T(1) + nu_));
    }

    T youngs_modulus() const override { return E_; }
    T shear_modulus() const override { return G_; }
    T poisson_ratio() const override { return nu_; }
    
    T section_area() const { return A_; }
    T moment_inertia() const { return I_; }
    T shear_coefficient() const { return k_; }

    T bending_stiffness() const override { return E_ * I_; }
    T shear_stiffness() const override { return k_ * G_ * A_; }

    std::string name() const override { return "SlenderBeam1d"; }

private:
    T E_;
    T nu_;
    T G_;
    T A_;
    T I_;
    T k_;
};
} // namespace pyck

#endif // PYCK_SLENDER_BEAM_1D_HPP
