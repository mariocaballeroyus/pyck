#include "tensor.hpp"
#include "bspline.hpp"
#include <tuple>
#include <utility>

namespace pyck
{

// === Constructors ===================================================================

template <std::floating_point T, std::size_t d>
TensorProduct<T, d>::TensorProduct(std::array<Ptr<const Basis<T>>, d> bases)
    : bases_(std::move(bases))
{
    for (std::size_t i = 0; i < d; ++i) {
        if (!bases_[i]) {
            throw std::invalid_argument("TensorProduct: "
                                        "Basis pointer is null.");
        }
    }
}

// === Properties =====================================================================

template <std::floating_point T, std::size_t d>
Index TensorProduct<T, d>::num_basis() const
{
    Index total = 1;
    for (std::size_t i = 0; i < d; ++i) {
        total *= bases_[i]->num_basis();
    }
    return total;
}

template <std::floating_point T, std::size_t d>
std::array<Index, d> TensorProduct<T, d>::num_intervals() const
{
    std::array<Index, d> intervals;
    for (std::size_t i = 0; i < d; ++i) {
        intervals[i] = bases_[i]->num_intervals();
    }
    return intervals;
}

template <std::floating_point T, std::size_t d>
const Basis<T>& TensorProduct<T, d>::basis(Index dir) const
{ 
    if (dir >= d) {
        throw std::out_of_range("Basis dimension index out of range");
    }
    return *bases_[dir]; 
}

// === Evaluation =====================================================================

template <std::floating_point T, std::size_t d>
void TensorProduct<T, d>::eval_on_span(const Matrix<T>& params,
                                       const std::array<Index, d>& spans,
                                       Index order,
                                       std::vector<Matrix<T>>& results) const
{
    if constexpr (d == 0) {
        return;
    }

    std::size_t num_points = params.rows();
    std::vector<Matrix<T>> acc_results;
    bases_[0]->eval_on_span(params.col(0), spans[0], order, acc_results);

    for (std::size_t i = 1; i < d; ++i)
    {
        std::vector<Matrix<T>> mat_i_derivs;
        bases_[i]->eval_on_span(params.col(i), spans[i], order, mat_i_derivs);

        std::vector<Matrix<T>> next_accumulated;
        next_accumulated.reserve(acc_results.size() * mat_i_derivs.size());

        for (const auto& res_mat : acc_results)
        {
            for (const auto& new_mat : mat_i_derivs)
            {
                std::size_t cols_res = res_mat.cols();
                std::size_t cols_mat = new_mat.cols();

                Matrix<T> combined(num_points, cols_res * cols_mat);

                for (std::size_t c1 = 0; c1 < cols_res; ++c1) {
                    for (std::size_t c2 = 0; c2 < cols_mat; ++c2) {
                        combined.col(c1 * cols_mat + c2) =
                            res_mat.col(c1).cwiseProduct(new_mat.col(c2));
                    }
                }
                next_accumulated.push_back(std::move(combined));
            }
        }
        acc_results = std::move(next_accumulated);
    }

    results = std::move(acc_results);
}

// === Template Instantiations ========================================================

template class TensorProduct<double, 1>;
template class TensorProduct<double, 2>;
template class TensorProduct<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class TensorProduct<float, 1>;
template class TensorProduct<float, 2>;
template class TensorProduct<float, 3>;
#endif

} // namespace pyck