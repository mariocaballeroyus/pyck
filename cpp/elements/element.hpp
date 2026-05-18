#ifndef PYCK_ELEMENT_HPP
#define PYCK_ELEMENT_HPP

#include <stdexcept>
#include <Eigen/Dense>

#include "patch.hpp"
#include "basis_derivs.hpp"
#include "intrinsic_geometry.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Base class for structural elements of parametric dimension d.
 *
 * @tparam T Scalar type.
 * @tparam d Parametric dimension (1 = beam, 2 = plate/shell).
 */
template <std::floating_point T, std::size_t d>
class Element
{
public:
    virtual ~Element() = default;

    // === DOF layout =================================================================

    /// @brief Number of DOFs per node.
    virtual std::size_t num_node_dofs() const = 0;

    /// @brief Minimum required derivative order for shape-function evaluation.
    virtual std::size_t min_order() const { return 0; }

    // === Matrix Operators (Element Formulation-Agnostic) ============================

    /**
     * @brief Generalised-stress shape matrix.
     *        S = D B.
     *
     * @param patch The patch.
     * @param basis The basis derivatives.
     * @param local The local geometry.
     * @return The generalised-stress shape matrix.
     */
    virtual Matrix<T> stress_matrix(const Patch<T, d>& patch,
                                    const BasisDerivs<T, d>& basis,
                                    const IntrinsicGeometry<T, d>& ig) const;

    /**
     * @brief Local stiffness matrix \
     *        K = \int_{\Omega}{ B^T D B dV }.
     *
     * @param patch The patch.
     * @param elem_idx The element index.
     * @param mapped_pts The mapped quadrature points.
     * @param q_weights The quadrature weights.
     * @param stiffness The local stiffness matrix.
     */
    virtual void compute_local_stiffness(const Patch<T, d>& patch,
                                         Index elem_idx,
                                         const ColMatrix<T, d>& mapped_pts,
                                         const Vector<T>& q_weights,
                                         Matrix<T>& stiffness) const;

    // === Matrix Operators (Element Formulation-Specific) ============================

    /**
     * @brief Strain-displacement operator B, row-stacked per qp.
     *
     * @param patch The patch.
     * @param basis The basis derivatives.
     * @param local The local geometry (provides metric, Christoffels, curvature).
     * @return The strain-displacement operator.
     */
    virtual Matrix<T> strain_matrix(const Patch<T, d>& patch,
                                    const BasisDerivs<T, d>& basis,
                                    const IntrinsicGeometry<T, d>& ig) const = 0;

    /**
     * @brief Constitutive operator D at quadrature point q.
     *
     * @param local The local geometry.
     * @param q Quadrature point index.
     * @return The constitutive operator.
     */
    virtual Matrix<T> constitutive_matrix(const IntrinsicGeometry<T, d>& ig,
                                          Index q) const = 0;

    // === Shape Matrices (Pure Virtual) ==============================================

    /**
     * @brief Transverse-displacement shape matrix N_w (Q × K).
     *
     * @param patch The patch.
     * @param basis The basis derivatives.
     * @param local The local geometry.
     * @return The transverse-displacement shape matrix.
     */
    virtual Matrix<T> displacement_shape_matrix(const Patch<T, d>& patch,
                                                const BasisDerivs<T, d>& basis,
                                                const IntrinsicGeometry<T, d>& ig) const = 0;

    /**
     * @brief Rotation shape matrix.
     *
     * @param patch The patch.
     * @param basis The basis derivatives.
     * @param local The local geometry.
     * @return The rotation shape matrix.
     */
    virtual Matrix<T> rotation_shape_matrix(const Patch<T, d>& patch,
                                            const BasisDerivs<T, d>& basis,
                                            const IntrinsicGeometry<T, d>& ig) const = 0;
};


template <std::floating_point T, std::size_t d>
Matrix<T>
Element<T, d>::stress_matrix(const Patch<T, d>& patch,
                             const BasisDerivs<T, d>& basis,
                             const IntrinsicGeometry<T, d>& ig) const
{
    const Matrix<T> B = strain_matrix(patch, basis, ig);
    const Index Q        = basis.N().rows();
    const Index n_strain = B.rows() / Q;
    const Index K        = B.cols();

    Matrix<T> Nsigma(n_strain * Q, K);
    for (Index q = 0; q < Q; ++q) {
        const Matrix<T> D = constitutive_matrix(ig, q);
        Nsigma.middleRows(n_strain * q, n_strain).noalias() =
            D * B.middleRows(n_strain * q, n_strain);
    }
    return Nsigma;
}

template <std::floating_point T, std::size_t d>
void
Element<T, d>::compute_local_stiffness(const Patch<T, d>& patch,
                                       Index elem_idx,
                                       const ColMatrix<T, d>& mapped_pts,
                                       const Vector<T>& q_weights,
                                       Matrix<T>& stiffness) const
{
    auto basis   = eval_basis(patch, mapped_pts, elem_idx, min_order());
    auto act_pts = patch.active_control_pts(elem_idx);
    IntrinsicGeometry ig(basis, act_pts);
    ig.compute_christoffels();

    const Matrix<T> B    = strain_matrix(patch, basis, ig);
    const Index Q        = q_weights.size();
    const Index n_strain = B.rows() / Q;
    const Index K        = B.cols();

    stiffness.setZero(K, K);
    for (Index q = 0; q < Q; ++q) {
        const T dV = q_weights(q) * ig.jac(q);
        const Matrix<T> D = constitutive_matrix(ig, q);
        const auto B_q = B.middleRows(n_strain * q, n_strain);
        stiffness.noalias() += dV * (B_q.transpose() * D * B_q);
    }
}

} // namespace pyck

#endif // PYCK_ELEMENT_HPP
