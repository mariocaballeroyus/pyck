#ifndef PYCK_LOAD_CONDITION_HPP
#define PYCK_LOAD_CONDITION_HPP

#include <vector>
#include <Eigen/Dense>

#include "condition.hpp"
#include "patch.hpp"
#include "quadrature.hpp"
#include "element.hpp"
#include "../types.hpp"

namespace pyck
{

/**
 * @brief Generic distributed domain load
 *
 * @tparam T Scalar type
 * @tparam d Parametric dimension
 */
template <std::floating_point T, std::size_t d>
class LoadCondition : public Condition<T, d>
{
public:

    // === Constructors ===============================================================

    LoadCondition(const Patch<T, d>& patch,
                  const Element<T, d>& element,
                  const QuadratureRule<T, d>& quadrature);

    // === Utility ====================================================================

    /// @brief Constant uniform load applied over the whole domain.
    LoadCondition& add(T value);

    /// @brief Pre-evaluated per-quadrature-point load values. Size must equal
    ///        num_active_qpts() (one value per active Gauss point).
    LoadCondition& add(const Vector<T>& values_at_qpts);

    void apply(SystemAssembler<T>& assembler,
               const DofLayout& layout,
               DofLayout::BlockId primal_block) const override;

    // === Properties =================================================================

    /// @brief The patch where the domain load is applied.
    const Patch<T, d>& patch() const override { return patch_; }

    /// @brief Number of active quadrature points (live elements × points/element).
    Index num_active_qpts() const;

private:

    struct Term {
        bool varying = false;       ///< True if the load varies per qp.
        T constant_value = T(0);    ///< Constant load (when not varying).
        Vector<T> values_at_qpts;   ///< Per-qp load; size num_active_qpts().
    };

    /// @brief Patch the load is integrated over.
    const Patch<T, d>& patch_;

    /// @brief Element formulation (supplies the displacement shape).
    const Element<T, d>& element_;

    /// @brief Domain quadrature rule.
    const QuadratureRule<T, d>& quadrature_;

    /// @brief Registered load terms.
    std::vector<Term> terms_;
};

} // namespace pyck

#endif // PYCK_LOAD_CONDITION_HPP