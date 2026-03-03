#include "tensor.hpp"
#include "bspline.hpp"
#include <tuple>
#include <utility>

namespace pyck
{


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

template <std::floating_point T, std::size_t d>
TensorProduct<T, d>::TensorProduct(Ptr<const Basis<T>> b0)
requires (d == 1)
    : bases_{std::move(b0)} 
{
    if (!bases_[0]) {
        throw std::invalid_argument("TensorProduct: "
                                    "Basis pointer is null.");
    }
}

template <std::floating_point T, std::size_t d>
TensorProduct<T, d>::TensorProduct(Ptr<const Basis<T>> b0, 
                                   Ptr<const Basis<T>> b1)
requires (d == 2)
    : bases_{std::move(b0), std::move(b1)} 
{
    if (!bases_[0] || !bases_[1]) {
        throw std::invalid_argument("TensorProduct: "
                                    "Basis pointer is null.");
    }
}

template <std::floating_point T, std::size_t d>
TensorProduct<T, d>::TensorProduct(Ptr<const Basis<T>> b0, 
                                   Ptr<const Basis<T>> b1, 
                                   Ptr<const Basis<T>> b2)
requires (d == 3)
    : bases_{std::move(b0), std::move(b1), std::move(b2)} 
{
    if (!bases_[0] || !bases_[1] || !bases_[2]) {
        throw std::invalid_argument("TensorProduct: Basis pointer is null.");
    }
}

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

template <std::floating_point T, std::size_t d>
Matrix<T> TensorProduct<T, d>::eval(const Matrix<T>& params,
                                    const std::array<Index, d>& spans) const
{
    if constexpr (d == 0) {
        return Matrix<T>(0, 0);
    }

    std::size_t num_points = params.rows();
    Matrix<T> result = bases_[0]->eval(params.col(0), spans[0]);

    for (std::size_t i = 1; i < d; ++i) {
        Matrix<T> mat_i = bases_[i]->eval(params.col(i), spans[i]);
        
        std::size_t cols_res = result.cols();
        std::size_t cols_mat = mat_i.cols();
        
        Matrix<T> next_result(num_points, cols_res * cols_mat);
        
        for (std::size_t c1 = 0; c1 < cols_res; ++c1) {
            for (std::size_t c2 = 0; c2 < cols_mat; ++c2) {
                next_result.col(c1 * cols_mat + c2) = result.col(c1).cwiseProduct(mat_i.col(c2));
            }
        }
        result = std::move(next_result);
    }
    
    return result;
}

template <std::floating_point T, std::size_t d>
std::vector<Matrix<T>> TensorProduct<T, d>::eval_derivs(const Matrix<T>& params,
                                                        const std::array<Index, d>& spans,
                                                        Index order) const
{
    if constexpr (d == 0) {
        return {};
    }

    std::size_t num_points = params.rows();
    std::vector<Matrix<T>> acc_results = bases_[0]->eval_derivs(params.col(0), spans[0], order);

    for (std::size_t i = 1; i < d; ++i) 
    {
        std::vector<Matrix<T>> mat_i_derivs = bases_[i]->eval_derivs(params.col(i), spans[i], order);

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

    return acc_results;
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