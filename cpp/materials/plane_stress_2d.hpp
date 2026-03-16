#ifndef PYCK_PLANE_STRESS_2D_HPP
#define PYCK_PLANE_STRESS_2D_HPP

#include "material.hpp"

namespace pyck
{

/**
 * @brief Isotropic linear elastic material model for 2D plane stress.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlaneStress2d : public Material<T, 2>
{
public:
    /**
     * @brief Construct a plane stress material.
     * @param E Young's modulus.
     * @param nu Poisson's ratio.
     * @param h Thickness.
     */
    PlaneStress2d(T E, T nu = T(0.3), T h = T(1.0), T k = T(5.0 / 6.0))
        : E_(E), nu_(nu), h_(h), k_(k)
    {
        if (E_ <= 0) {
            throw std::invalid_argument("PlaneStress2d: E must be positive.");
        }
        if (nu_ <= -1.0 || nu_ >= 0.5) {
            throw std::invalid_argument("PlaneStress2d: nu must be in (-1, 0.5).");
        }
        if (h_ <= 0) {
            throw std::invalid_argument("PlaneStress2d: h must be positive.");
        }
        if (k_ <= 0) {
            throw std::invalid_argument("PlaneStress2d: k must be positive.");
        }
    }

    T youngs_modulus() const override { return E_; }
    T poisson_ratio() const override { return nu_; }
    T shear_modulus() const override { return E_ / (T(2) * (T(1) + nu_)); }
    T thickness() const { return h_; }

    T bending_stiffness() const override { return (E_ * h_ * h_ * h_) / (T(12) * (T(1) - nu_ * nu_)); }
    T shear_stiffness() const override { return k_ * shear_modulus() * h_; }
    T shear_correction_factor() const { return k_; }

    std::string name() const override { return "PlaneStress2d"; }

    /**
     * @brief Compute the 3x3 plane stress constitutive matrix.
     * 
     * The matrix C relates stresses to strains:
     * [sigma_xx, sigma_yy, tau_xy]^T = C * [epsilon_xx, epsilon_yy, gamma_xy]^T
     * 
     * For an isotropic material, C is:
     * [ C11  C12   0  ]
     * [ C21  C22   0  ]
     * [  0    0   C33 ]
     * 
     * where:
     * C11 = C22 = E / (1 - nu^2)
     * C12 = C21 = E * nu / (1 - nu^2)
     * C33 = G = E / (2 * (1 + nu))
     */
    Matrix<T> constitutive_matrix() const override
    {
        Matrix<T> C = Matrix<T>::Zero(3, 3);
        T factor = E_ / (T(1) - nu_ * nu_);
        C(0, 0) = factor;
        C(0, 1) = factor * nu_;
        C(1, 0) = factor * nu_;
        C(1, 1) = factor;
        C(2, 2) = factor * (T(1) - nu_) / T(2);
        return C;
    }

    /**
     * @brief Get the 3x3 bending constitutive matrix D_b.
     */
    Matrix<T> bending_matrix() const
    {
        Matrix<T> Db = Matrix<T>::Zero(3, 3);
        T D = bending_stiffness();
        Db(0, 0) = D;
        Db(0, 1) = D * nu_;
        Db(1, 0) = D * nu_;
        Db(1, 1) = D;
        Db(2, 2) = D * (T(1) - nu_) / T(2);
        return Db;
    }

    /**
     * @brief Get the 2x2 shear constitutive matrix D_s.
     */
    Matrix<T> shear_matrix() const
    {
        return Matrix<T>::Identity(2, 2) * shear_stiffness();
    }

private:
    T E_;
    T nu_;
    T h_;
    T k_;
};

} // namespace pyck

#endif // PYCK_PLANE_STRESS_2D_HPP
