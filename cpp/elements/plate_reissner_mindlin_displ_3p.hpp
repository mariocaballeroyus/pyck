#ifndef PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP
#define PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP

#include <stdexcept>
#include <vector>

#include "element.hpp"
#include "../materials/plane_stress_2d.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Split-displacement Reissner-Mindlin plate element.
 *
 * Primary variables per node:
 *   - w_b  : bending contribution to the transverse displacement
 *   - w_s1 : shear contribution coupled to gamma_x = w_s1,x
 *   - w_s2 : shear contribution coupled to gamma_y = w_s2,y
 *
 * Recovered fields:
 *   w     = w_b + w_s1 + w_s2
 *   phi_x = -w_b,x - w_s2,x
 *   phi_y = -w_b,y - w_s1,y
 *   gamma = [w_s1,x, w_s2,y]^T
 *   kappa = L phi
 *         = [-w_b,xx - w_s2,xx,
 *            -w_b,yy - w_s1,yy,
 *            -2 w_b,xy - w_s1,xy - w_s2,xy]^T
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlinDispl3p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    explicit PlateReissnerMindlinDispl3p(Ptr<PlaneStress2d<T>> material)
        : material_(material)
    {
        if (!material_) {
            throw std::invalid_argument(
                "PlateReissnerMindlinDispl3p: material is null."
            );
        }
    }

    Matrix<T> bending_constitutive_matrix() const override { return material_->bending_matrix(); }
    Matrix<T> shear_constitutive_matrix() const override { return material_->shear_matrix(); }

    Matrix<T> displacement_shape_matrix(
        const std::vector<Matrix<T>>& shape_derivs) const override
    {
        const auto& N = shape_derivs;
        const Index Q = N[idx::val].rows();
        const Index n = N[idx::val].cols();
        Matrix<T> Nw = Matrix<T>::Zero(Q, 3 * n);

        for (Index i = 0; i < n; ++i) {
            Nw.col(3 * i    ) = N[idx::val].col(i);
            Nw.col(3 * i + 1) = N[idx::val].col(i);
            Nw.col(3 * i + 2) = N[idx::val].col(i);
        }
        return Nw;
    }

    Matrix<T> rotation_shape_matrix(
        const std::vector<Matrix<T>>& shape_derivs) const override
    {
        const auto& N = shape_derivs;
        const Index Q = N[idx::val].rows();
        const Index n = N[idx::val].cols();
        Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 3 * n);

        for (Index q = 0; q < Q; ++q) {
            for (Index i = 0; i < n; ++i) {
                Nphi(2 * q,     3 * i)     = -N[idx::d1](q, i);
                Nphi(2 * q,     3 * i + 2) = -N[idx::d1](q, i);
                Nphi(2 * q + 1, 3 * i)     = -N[idx::d2](q, i);
                Nphi(2 * q + 1, 3 * i + 1) = -N[idx::d2](q, i);
            }
        }
        return Nphi;
    }

    Matrix<T> bending_strain_matrix(
        const std::vector<Matrix<T>>& shape_derivs) const override
    {
        const auto& N = shape_derivs;
        const Index Q = N[idx::val].rows();
        const Index n = N[idx::val].cols();
        Matrix<T> Bb = Matrix<T>::Zero(3 * Q, 3 * n);

        for (Index q = 0; q < Q; ++q) {
            for (Index i = 0; i < n; ++i) {
                Bb(3 * q,     3 * i)     = -N[idx::d11](q, i);
                Bb(3 * q,     3 * i + 2) = -N[idx::d11](q, i);

                Bb(3 * q + 1, 3 * i)     = -N[idx::d22](q, i);
                Bb(3 * q + 1, 3 * i + 1) = -N[idx::d22](q, i);

                Bb(3 * q + 2, 3 * i)     = -T(2) * N[idx::d12](q, i);
                Bb(3 * q + 2, 3 * i + 1) = -N[idx::d12](q, i);
                Bb(3 * q + 2, 3 * i + 2) = -N[idx::d12](q, i);
            }
        }
        return Bb;
    }

    Matrix<T> shear_strain_matrix(
        const std::vector<Matrix<T>>& shape_derivs) const override
    {
        const auto& N = shape_derivs;
        const Index Q = N[idx::val].rows();
        const Index n = N[idx::val].cols();
        Matrix<T> Bs = Matrix<T>::Zero(2 * Q, 3 * n);

        for (Index q = 0; q < Q; ++q) {
            for (Index i = 0; i < n; ++i) {
                Bs(2 * q,     3 * i + 1) = N[idx::d1](q, i);
                Bs(2 * q + 1, 3 * i + 2) = N[idx::d2](q, i);
            }
        }
        return Bs;
    }

    std::size_t num_node_dofs() const override { return 3; }

    std::size_t min_order() const override { return 2; }

    std::array<std::size_t, 2> rotation_dof_indices() const override
    {
        return {0, 0};
    }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_3P_HPP
