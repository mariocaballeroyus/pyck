#ifndef PYCK_KIRCHHOFF_LOVE_PLATE_HPP
#define PYCK_KIRCHHOFF_LOVE_PLATE_HPP

#include <vector>

#include "element.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Kirchhoff-Love thin plate element.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class KirchhoffLovePlate : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    /**
     * @brief Construct a Kirchhoff-Love plate element.
     * @param youngs_modulus Young's modulus E.
     * @param poisson_ratio Poisson's ratio ν.
     * @param thickness     Plate thickness h.
     */
    KirchhoffLovePlate(T youngs_modulus,
                       T poisson_ratio,
                       T thickness);

    /**
     * @brief Compute the local stiffness matrix for the element.
     * @param patch The patch of the element.
     * @param q_points Quadrature points.
     * @param q_weights Quadrature weights.
     * @param span Knot-span index.
     * @param stiffness The local stiffness matrix to be computed.
     */
    void compute_local_stiffness(const Patch<T, 2>& patch,
                                 const ColMatrix<T, 2>& q_points,
                                 const Vector<T>& q_weights,
                                 Index span,
                                 Matrix<T>& stiffness) const override;

    /**
     * @brief Compute the generalized shape function matrix N.
     * @param shape_derivs Pre-evaluated shape functions and their derivatives.
     */
    Matrix<T> shape_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /**
     * @brief Compute the generalized strain-displacement matrix B.
     * @param shape_derivs Pre-evaluated shape functions and their derivatives.
     */
    Matrix<T> strain_displacement_matrix(const std::vector<Matrix<T>>& shape_derivs) const override;

    /// @brief Get the number of degrees of freedom per node.
    std::size_t num_dofs_per_node() const override { return 1; }

    /// @brief Get the minimum required derivative order for shape_matrix evaluation.
    std::size_t required_shape_order() const override { return 0; }

private:
    /// @brief Material Parameters
    T E_, nu_, h_, Kb_;

    /// @brief Pre-computed 3×3 bending constitutive matrix.
    Matrix<T> Kb_mat_;
};

} // namespace pyck

#endif // PYCK_KIRCHHOFF_LOVE_PLATE_HPP
