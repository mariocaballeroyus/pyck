#ifndef PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Two-parameter rotation-free Reissner-Mindlin plate element via Helmholtz 
 *        decomposition.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlinDispl2p : public Element<T, 2>
{
public:

    // === Constructors ===========================================================

    /**
     * @brief Constructor.
     *
     * @param material Material properties.
     */
    explicit PlateReissnerMindlinDispl2p(Ptr<PlaneStress2d<T>> material);

    // === Matrix Operators =======================================================

    /**
     * @brief Strain-displacement matrix B (5Q × 2n). Five strain rows per qp:
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

    /// @brief Number of node degrees of freedom (w_b + ψ).
    std::size_t num_node_dofs() const override
    { return 2; }
    Index basis_order() const override { return 3; }

    unsigned flags() const override
    { return Flags::Metric | Flags::KernelHessian | Flags::KernelLB | Flags::KernelLBGradient; }

    unsigned essential_flags() const override
    { return Flags::Metric | Flags::KernelHessian | Flags::KernelLB | Flags::KernelLBGradient; }
    unsigned natural_flags()   const override
    { return Flags::Metric | Flags::KernelHessian | Flags::KernelLB | Flags::KernelLBGradient; }

private:

    /// @brief Material properties.
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP
