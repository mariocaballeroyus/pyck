#ifndef PYCK_PLATE_REISSNER_MINDLIN_1P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_1P_HPP

#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Single-variable Reissner-Mindlin plate element.
 *
 * Uses a bending potential w_b as the sole primary variable.
 * The total deflection is recovered via a Laplacian correction:
 *   w = w_b - (K_b / K_s) * Laplacian(w_b)
 *
 * Requires at least C^2 continuity (cubic B-splines or higher)
 * to satisfy the third-order derivative requirements.
 *
 * DOF per node (1):
 *   - w_b: Bending potential
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlin1p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:

    /**
     * @brief Construct a Reissner-Mindlin plate element with the given material.
     * 
     * @param material  Material model providing bending and shear stiffnesses.
     */
    PlateReissnerMindlin1p(Ptr<PlaneStress2d<T>> material);

    void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                 const Vector<T>& jacobian,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    /// @brief Effective transverse displacement shape functions Ñ_w (Q × n).
    ///
    /// Accounts for the Laplacian correction w = w_b - (K_b/K_s) Δw_b, so
    /// this is: Ñ_i = N_i - (K_b/K_s)(N_i,xx + N_i,yy).
    Matrix<T> displacement_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Rotation shape matrix (2Q × n): row `2q` = −N,x, row `2q+1` = −N,y.
    ///
    /// RM-1p ansatz gives θ = -∇w_b (the shear-correction term in
    /// w = w_b - ratio·Δw_b cancels against -γ in θ = γ - ∇w).
    /// Both components act on the sole w_b DOF.
    Matrix<T> rotation_shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    std::size_t num_node_dofs() const override { return 1; }

    std::size_t min_order() const override { return 3; }

    std::array<std::size_t, 2> rotation_dof_indices() const override { return {0, 0}; }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_1P_HPP
