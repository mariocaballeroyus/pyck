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
 * @brief Two-parameter rotation-free Reissner-Mindlin plate element.
 *
 * Primary variables per node:
 *   - w_b : bending contribution to the transverse displacement
 *   - psi : scalar Helmholtz potential for the curl part of the rotation
 *
 * Recovered fields (with K_b = bending stiffness, K_s = shear stiffness):
 *   w     = w_b - (K_b/K_s) * Laplacian(w_b)
 *   phi_x = -w_b,x + psi,y
 *   phi_y = -w_b,y - psi,x
 *   gamma = [ -(K_b/K_s)(w_b,xxx + w_b,xyy) + psi,y,
 *             -(K_b/K_s)(w_b,xxy + w_b,yyy) - psi,x ]^T
 *   kappa = L phi
 *         = [ -w_b,xx + psi,xy,
 *             -w_b,yy - psi,xy,
 *             -2 w_b,xy + psi,yy - psi,xx ]^T
 *
 * Requires at least C^2 continuity (cubic B-splines or higher) due to the
 * third-order derivatives of w_b in the shear strain.
 *
 * @tparam T Scalar type.
 */
template <std::floating_point T>
class PlateReissnerMindlinDispl2p : public Element<T, 2>
{
    using idx = typename Element<T, 2>::idx;

public:
    explicit PlateReissnerMindlinDispl2p(Ptr<PlaneStress2d<T>> material)
        : material_(material)
    {
        if (!material_) {
            throw std::invalid_argument(
                "PlateReissnerMindlinDispl2p: material is null."
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
        const T ratio = material_->bending_stiffness()
                      / material_->shear_stiffness();

        // Nw_i = [ N_i - (Kb/Ks)(N_i,xx + N_i,yy)   0 ]
        Matrix<T> Nw = Matrix<T>::Zero(Q, 2 * n);
        for (Index i = 0; i < n; ++i) {
            Nw.col(2 * i) = N[idx::val].col(i)
                          - ratio * (N[idx::d11].col(i) + N[idx::d22].col(i));
        }
        return Nw;
    }

    Matrix<T> rotation_shape_matrix(
        const std::vector<Matrix<T>>& shape_derivs) const override
    {
        const auto& N = shape_derivs;
        const Index Q = N[idx::val].rows();
        const Index n = N[idx::val].cols();
        Matrix<T> Nphi = Matrix<T>::Zero(2 * Q, 2 * n);

        // Nphi_i = [ -N_i,x   N_i,y
        //           -N_i,y  -N_i,x ]
        for (Index q = 0; q < Q; ++q) {
            for (Index i = 0; i < n; ++i) {
                Nphi(2 * q,     2 * i)     = -N[idx::d1](q, i);
                Nphi(2 * q,     2 * i + 1) =  N[idx::d2](q, i);
                Nphi(2 * q + 1, 2 * i)     = -N[idx::d2](q, i);
                Nphi(2 * q + 1, 2 * i + 1) = -N[idx::d1](q, i);
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
        const T ratio = material_->bending_stiffness()
                      / material_->shear_stiffness();

        // Bb_i (3 x 2): bending strains kappa = L phi
        //   row 0 (kappa_x):    [ -N_i,xx              N_i,xy             ]
        //   row 1 (kappa_y):    [ -N_i,yy             -N_i,xy             ]
        //   row 2 (2 kappa_xy): [ -2 N_i,xy            N_i,yy - N_i,xx    ]
        // Bs_i (2 x 2): shear strains gamma = -(Kb/Ks) grad(Lap w_b) + curl psi
        //   row 0 (gamma_x): [ -(Kb/Ks)(N_i,xxx + N_i,xyy)   N_i,y ]
        //   row 1 (gamma_y): [ -(Kb/Ks)(N_i,xxy + N_i,yyy)  -N_i,x ]
        Matrix<T> B = Matrix<T>::Zero(5 * Q, 2 * n);
        for (Index q = 0; q < Q; ++q) {
            for (Index i = 0; i < n; ++i) {
                // Bending block
                B(5 * q,     2 * i)     = -N[idx::d11](q, i);
                B(5 * q,     2 * i + 1) =  N[idx::d12](q, i);

                B(5 * q + 1, 2 * i)     = -N[idx::d22](q, i);
                B(5 * q + 1, 2 * i + 1) = -N[idx::d12](q, i);

                B(5 * q + 2, 2 * i)     = -T(2) * N[idx::d12](q, i);
                B(5 * q + 2, 2 * i + 1) =  N[idx::d22](q, i) - N[idx::d11](q, i);

                // Shear block
                B(5 * q + 3, 2 * i)     = -ratio * (N[idx::d111](q, i)
                                                  + N[idx::d122](q, i));
                B(5 * q + 3, 2 * i + 1) =  N[idx::d2](q, i);

                B(5 * q + 4, 2 * i)     = -ratio * (N[idx::d112](q, i)
                                                  + N[idx::d222](q, i));
                B(5 * q + 4, 2 * i + 1) = -N[idx::d1](q, i);
            }
        }
        return B;
    }

    std::size_t num_node_dofs() const override { return 2; }

    std::size_t min_order() const override { return 3; }

    std::array<std::size_t, 2> rotation_dof_indices() const override
    {
        return {0, 0};
    }

private:
    Ptr<PlaneStress2d<T>> material_;
};

} // namespace pyck

#endif // PYCK_PLATE_REISSNER_MINDLIN_DISPL_2P_HPP
