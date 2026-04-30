#ifndef PYCK_BOUNDARY_FIELD_HPP
#define PYCK_BOUNDARY_FIELD_HPP

#include <concepts>
#include <vector>

#include "../elements/element.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Polymorphic boundary-field operator.
 *
 * A BoundaryField builds the Q×K constraint-row matrix that picks out a
 * scalar quantity (e.g. transverse displacement, normal rotation) from the
 * element's shape functions at each quadrature point on the boundary.
 */
template <std::floating_point T>
class BoundaryField
{
public:
    virtual ~BoundaryField() = default;

    /**
     * @brief Build the Q×K constraint-row matrix at each quadrature point.
     */
    virtual Matrix<T> evaluate(
        const Element<T, 2>& element,
        const std::vector<Matrix<T>>& sfd,
        const ColMatrix<T, 3>& outward_normal) const = 0;
};

/**
 * @brief Transverse displacement w.
 */
template <std::floating_point T>
class TransverseDisplacement : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(const Element<T, 2>& element,
                       const std::vector<Matrix<T>>& sfd,
                       const ColMatrix<T, 3>& /*outward_normal*/) const override
    {
        return element.displacement_shape_matrix(sfd);
    }
};

/**
 * @brief Rotation projected onto the outward boundary normal: θ_n = n·θ.
 */
template <std::floating_point T>
class NormalRotation : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(const Element<T, 2>& element,
                       const std::vector<Matrix<T>>& sfd,
                       const ColMatrix<T, 3>& outward_normal) const override
    {
        const Matrix<T> Nrot = element.rotation_shape_matrix(sfd);
        const Index Q = static_cast<Index>(outward_normal.rows());
        Matrix<T> C(Q, Nrot.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n_x = outward_normal(q, 0);
            const T n_y = outward_normal(q, 1);
            C.row(q) = n_x * Nrot.row(2 * q) + n_y * Nrot.row(2 * q + 1);
        }
        return C;
    }
};

/**
 * @brief Rotation projected onto the boundary tangent: θ_s = s·θ, with
 *        s = (-n_y, n_x) the in-plane 90° rotation of the outward normal.
 */
template <std::floating_point T>
class TangentialRotation : public BoundaryField<T>
{
public:
    Matrix<T> evaluate(const Element<T, 2>& element,
                       const std::vector<Matrix<T>>& sfd,
                       const ColMatrix<T, 3>& outward_normal) const override
    {
        const Matrix<T> Nrot = element.rotation_shape_matrix(sfd);
        const Index Q = static_cast<Index>(outward_normal.rows());
        Matrix<T> C(Q, Nrot.cols());
        for (Index q = 0; q < Q; ++q) {
            const T n_x = outward_normal(q, 0);
            const T n_y = outward_normal(q, 1);
            C.row(q) = -n_y * Nrot.row(2 * q) + n_x * Nrot.row(2 * q + 1);
        }
        return C;
    }
};

} // namespace pyck

#endif // PYCK_BOUNDARY_FIELD_HPP
