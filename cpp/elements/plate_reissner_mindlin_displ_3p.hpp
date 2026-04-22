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
 * This preserves the displacement-only nature of the formulation while
 * keeping the split transverse fields kinematically consistent with both
 * gamma = grad(w) + phi and kappa = L phi.
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

    void compute_local_stiffness(const std::vector<Matrix<T>>& shape_fns,
                                 const Vector<T>& jacobian,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override
    {
        Matrix<T> B = this->strain_displacement_matrix(shape_fns);
        const auto Db = material_->bending_matrix();
        const auto Ds = material_->shear_matrix();

        const Index Q = q_weights.size();
        const Index K = B.cols();
        const Vector<T> dV = q_weights.cwiseProduct(jacobian);

        stiffness.setZero(K, K);
        for (Index q = 0; q < Q; ++q) {
            const auto Bb = B.middleRows(5 * q, 3);
            const auto Bs = B.middleRows(5 * q + 3, 2);
            stiffness.noalias() += dV[q] * (
                Bb.transpose() * Db * Bb
                + Bs.transpose() * Ds * Bs
            );
        }
    }

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
                Nphi(2 * q,     3 * i) = -N[idx::d1](q, i);
                Nphi(2 * q,     3 * i + 2) = -N[idx::d1](q, i);
                Nphi(2 * q + 1, 3 * i) = -N[idx::d2](q, i);
                Nphi(2 * q + 1, 3 * i + 1) = -N[idx::d2](q, i);
            }
        }
        return Nphi;
    }

    Matrix<T> strain_displacement_matrix(
        const std::vector<Matrix<T>>& shape_derivs) const override
    {
        const auto& N = shape_derivs;
        const Index Q = N[idx::val].rows();
        const Index n = N[idx::val].cols();
        Matrix<T> B = Matrix<T>::Zero(5 * Q, 3 * n);

        for (Index q = 0; q < Q; ++q) {
            for (Index i = 0; i < n; ++i) {
                B(5 * q,     3 * i) = -N[idx::d11](q, i);
                B(5 * q,     3 * i + 2) = -N[idx::d11](q, i);

                B(5 * q + 1, 3 * i) = -N[idx::d22](q, i);
                B(5 * q + 1, 3 * i + 1) = -N[idx::d22](q, i);

                B(5 * q + 2, 3 * i) = -T(2) * N[idx::d12](q, i);
                B(5 * q + 2, 3 * i + 1) = -N[idx::d12](q, i);
                B(5 * q + 2, 3 * i + 2) = -N[idx::d12](q, i);

                B(5 * q + 3, 3 * i + 1) = N[idx::d1](q, i);
                B(5 * q + 4, 3 * i + 2) = N[idx::d2](q, i);
            }
        }
        return B;
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
