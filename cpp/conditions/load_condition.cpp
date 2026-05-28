#include "load_condition.hpp"

#include "tensor_product.hpp"
#include "element_values.hpp"

#include <stdexcept>
#include <string>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
LoadCondition<T, d>::LoadCondition(const Patch<T, d>& patch,
                                   const Element<T, d>& element,
                                   const QuadratureRule<T, d>& quadrature)
    : patch_(patch),
      element_(element),
      quadrature_(quadrature)
{}

// === Utility ========================================================================

template <std::floating_point T, std::size_t d>
LoadCondition<T, d>& LoadCondition<T, d>::add(T value)
{
    Term term;
    term.varying = false;
    term.constant_value = value;
    terms_.push_back(std::move(term));
    return *this;
}

template <std::floating_point T, std::size_t d>
LoadCondition<T, d>& LoadCondition<T, d>::add(const Vector<T>& values_at_qpts)
{
    const Index nq = num_active_qpts();
    if (values_at_qpts.size() != static_cast<Eigen::Index>(nq)) {
        throw std::runtime_error("LoadCondition::add: values_at_qpts has size " +
                                 std::to_string(values_at_qpts.size()) +
                                 " but expected " + std::to_string(nq));
    }
    Term term;
    term.varying = true;
    term.values_at_qpts = values_at_qpts;
    terms_.push_back(std::move(term));
    return *this;
}

template <std::floating_point T, std::size_t d>
Index LoadCondition<T, d>::num_active_qpts() const
{
    return patch_.tensor_product().num_elements()
         * static_cast<Index>(quadrature_.num_points());
}

// === Assembly =======================================================================

template <std::floating_point T, std::size_t d>
void LoadCondition<T, d>::apply(SystemAssembler<T>& assembler,
                                const DofLayout& layout,
                                DofLayout::BlockId primal_block) const
{
    if (terms_.empty()) return;


    // Get the number of elements, primal DOFs and Gauss points
    ElementValues<T, d> ev(patch_, element_.basis_order(), element_.flags(), quadrature_);
    const Index n_elems  = ev.num_elements();
    const Index n_cps    = static_cast<Index>(ev.act_pts_.rows());
    const Index n_primal = n_cps * static_cast<Index>(element_.num_node_dofs());
    const Index n_gp = static_cast<Index>(quadrature_.num_points());

    // --- Assemble Local Blocks ------------------------------------------------------
    // f_load = \int{ q N_w^T d\Omega }

    // Preallocate local load vector
    Vector<T> f_local(n_primal);

    // Loop over elements
    std::size_t qpt_offset = 0;
    for (Index e = 0; e < n_elems; ++e) {
        // Compute values at the element
        ev.reinit(e);
        element_.displacement_shape_matrix(ev);
        const Matrix<T>& N_w = element_.N_w_workspace_;

        // Reset the reused local load vector (all terms accumulate per element)
        f_local.setZero();

        // Loop over load terms
        for (const auto& term : terms_) {
            // Loop over quadrature points
            for (Index q = 0; q < n_gp; ++q) {
                // Prescribed load: constant or per-qp value
                const T q_val = term.varying
                    ? term.values_at_qpts(static_cast<Eigen::Index>(qpt_offset + q))
                    : term.constant_value;
                // Domain integration measure
                const T dV = ev.mapped_weights_(q) * ev.jac(q);
                // Add contribution to the local load vector
                f_local.noalias() += (q_val * dV) * N_w.row(q).transpose();
            }
        }
        qpt_offset += static_cast<std::size_t>(n_gp);

        // --- Scatter into Global System -------------------------------------------------
        // {f + f_load}

        // Scatter the element primal DOFs
        layout.scatter_primal(primal_block, ev.elem_cps_, ev.elem_dofs_);
        for (Index i = 0; i < n_primal; ++i) {
            assembler.add_load(ev.elem_dofs_[i], f_local(i));
        }
    }
}

// === Template Instantiations ========================================================

template class LoadCondition<double, 1>;
template class LoadCondition<double, 2>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class LoadCondition<float, 1>;
template class LoadCondition<float, 2>;
#endif

} // namespace pyck
