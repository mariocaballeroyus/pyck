#ifndef PYCK_MATERIAL_HPP
#define PYCK_MATERIAL_HPP

#include <concepts>
#include <string>
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Base class for material models.
 * @tparam T Scalar type.
 * @tparam Dim Dimension of the problem (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
 */
template <typename T, int Dim>
class Material
{
public:
    virtual ~Material() = default;

    /**
     * @brief Get the name of the material law.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get the Young's modulus E.
     */
    virtual T youngs_modulus() const = 0;

    /**
     * @brief Get the Poisson's ratio ν.
     */
    virtual T poisson_ratio() const = 0;

    /**
     * @brief Get the shear modulus G.
     */
    virtual T shear_modulus() const = 0;

    /**
     * @brief Get the bending stiffness (EI for beams, D for plates).
     */
    virtual T bending_stiffness() const = 0;

    /**
     * @brief Get the shear stiffness (kGA for beams, kGh for plates).
     */
    virtual T shear_stiffness() const = 0;

    /**
     * @brief Get the constitutive matrix for the given dimension.
     */
    virtual Matrix<T> constitutive_matrix() const {
        throw std::runtime_error("This material does not support a constitutive matrix.");
    }
};

} // namespace pyck

#endif // PYCK_MATERIAL_HPP
