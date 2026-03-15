#ifndef PYCK_BEAM_MATERIAL_HPP
#define PYCK_BEAM_MATERIAL_HPP

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
class BeamMaterial : public Material<T, 1>
{
public:
    /**
     * @brief Construct a BeamMaterial.
     * @param E Young's modulus.
     * @param A Cross-sectional area.
     * @param I Second moment of area (moment of inertia).
     * @param G Shear modulus (optional, defaults to isotropic E/2.6).
     * @param k Shear correction factor (optional, defaults to 5/6).
     */
    BeamMaterial(T E, T A, T I, T G = T(-1), T k = T(5.0 / 6.0))
        : E_(E), A_(A), I_(I), k_(k)
    {
        if (E_ <= 0) throw std::invalid_argument("BeamMaterial: 
                                                 E must be positive.");
        if (A_ <= 0) throw std::invalid_argument("BeamMaterial: 
                                                 A must be positive.");
        if (I_ <= 0) throw std::invalid_argument("BeamMaterial: 
                                                 I must be positive.");
        
        if (G <= 0) {
            G_ = E_ / (T(2) * (T(1) + T(0.3)));
        } else {
            G_ = G;
        }
    }

    T youngs_modulus() const override { return E_; }
    T shear_modulus() const override { return G_; }
    T poisson_ratio() const override { return (E_ / (T(2) * G_)) - T(1); }
    
    T section_area() const { return A_; }
    T moment_inertia() const { return I_; }
    T shear_coefficient() const { return k_; }

    std::string name() const override { return "BeamMaterial"; }

private:
    T E_;
    T G_;
    T A_;
    T I_;
    T k_;
};

} // namespace pyck

#endif // PYCK_BEAM_MATERIAL_HPP
