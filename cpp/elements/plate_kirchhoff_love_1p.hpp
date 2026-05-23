#ifndef PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
#define PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Standard Kirchhoff-Love thin plate element.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateKirchhoffLove1p : public Element<T, 2>
{
public:

    // === Constructors ===========================================================

    /**
     * @brief Constructor.
     * 
     * @param material Material properties.
     */
    PlateKirchhoffLove1p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators ===========================================================

    /**
     * @brief Strain-displacement matrix B (3Q × K).
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return B-matrix.
     */
    void strain_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Constitutive D-matrix.
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return D-matrix.
     */
    ConstitutiveMatrix<T> constitutive_matrix(const ElementValues<T, 2>& ev, Index q) const override;

    // === Shape Matrices =============================================================

    /**
     * @brief Displacement shape matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Displacement shape matrix.
     */
    void displacement_shape_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Rotation shape matrix.
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Rotation shape matrix.
     */
    void rotation_shape_matrix(const ElementValues<T, 2>& ev) const override;

    // === Getters ================================================================

    /// @brief Number of node degrees of freedom (displacement + rotation).
    std::size_t num_node_dofs() const override 
    { return 1; }
    Index basis_order() const override { return 2; }

    unsigned flags() const override { return Flags::Metric | Flags::Christoffels; }

private:

    /// @brief Material properties.
    Ptr<PlaneStress2d<T>> material_;

};

} // namespace pyck

#endif // PYCK_PLATE_KIRCHHOFF_LOVE_1P_HPP
