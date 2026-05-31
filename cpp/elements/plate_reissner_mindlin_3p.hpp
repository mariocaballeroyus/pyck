#ifndef PYCK_REISSNER_MINDLIN_PLATE_3P_HPP
#define PYCK_REISSNER_MINDLIN_PLATE_3P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Standard Reissner-Mindlin plate element.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlin3p : public Element<T, 2>
{
public:

    // === Constructor ============================================================

    /**
     * @brief Constructor.
     * 
     * @param material Material properties.
     */
    PlateReissnerMindlin3p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators =======================================================

    /**
     * @brief Strain-displacement matrix B (5Q × 3n). Five strain rows per qp:
     *   rows 5q..5q+2 : curvatures κ_{11}, κ_{22}, 2κ_{12}
     *   rows 5q+3..5q+4 : transverse shears γ_1, γ_2
     *
     * @param patch Patch.
     * @param basis Basis derivatives.
     * @param local Local frame.
     * @param chr Christoffel symbols.
     * @return Strain-displacement matrix.
     */
    void strain_matrix(const ElementValues<T, 2>& ev) const override;

    /**
     * @brief Constitutive D-matrix (5×5 block-diag [Db; Ds]).
     *
     * @param local Local frame.
     * @param q Quadrature point.
     * @return Constitutive matrix.
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
    std::size_t num_node_dofs() const override { return 3; }

    Index basis_order() const override { return 2; }

    unsigned flags() const override { return Flags::Metric; }

    unsigned essential_flags() const override { return Flags::None; }
    unsigned natural_flags()   const override { return Flags::Metric; }

private:

    /// @brief Material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_REISSNER_MINDLIN_PLATE_3P_HPP
