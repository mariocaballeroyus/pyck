#ifndef PYCK_EULER_BERNOULLI_BEAM_1P_HPP
#define PYCK_EULER_BERNOULLI_BEAM_1P_HPP

#include <vector>

#include "element.hpp"
#include "../types.hpp"

namespace pyck
{

/// @brief Euler-Bernoulli beam element with one quadrature point.
/// @tparam T Scalar type.
template <std::floating_point T>
class EulerBernoulliBeam1P : public Element<T, 1>
{
public:

    using ScalarType = T;

    // === Constructors ===============================================================

    EulerBernoulliBeam1P(ScalarType E, ScalarType A, ScalarType I)
    : E_(E), A_(A), I_(I), Kb_(E * I) {}

    // === Local Element Assemblies ===================================================

    /**
     * @brief Compute the local stiffness matrix for the element.
     * @param patch The patch of the element.
     * @param quadrature The quadrature rule for the element.
     * @param stiffness The local stiffness matrix to be computed.
     */
    void compute_local_stiffness(const Patch<T, 1>& patch,
                                 const ColMatrix<T, 1>& q_points,
                                 const Vector<T>& q_weights,
                                 Matrix<T>& stiffness) const override;

    /**
     * @brief Compute the local load vector for the element.
     * @param patch The patch of the element.
     * @param quadrature The quadrature rule for the element.
     * @param load The local load vector to be computed.
     */
    void compute_local_load(const Patch<T, 1>& patch,
                            const ColMatrix<T, 1>& q_points,
                            const Vector<T>& q_weights,
                            const Vector<T>& load_values,
                            Vector<T>& load) const override;

private:

    /// @brief Young's Modulus of the beam material
    ScalarType E_;

    /// @brief Area moment of inertia of the beam cross-section
    ScalarType I_;

    /// @brief Cross-sectional area of the beam
    ScalarType A_;
    
    /// @brief Bending stiffness of the beam (E * I)
    ScalarType Kb_;
};

} // namespace pyck

#endif // PYCK_EULER_BERNOULLI_BEAM_1P_HPP
