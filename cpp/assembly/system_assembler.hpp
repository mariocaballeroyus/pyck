#ifndef PYCK_SYSTEM_ASSEMBLER_HPP
#define PYCK_SYSTEM_ASSEMBLER_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "../types.hpp"

namespace pyck
{

/**
 * @brief Sparse-assembly sink for the global system.
 *
 *        Accumulates global stiffness contributions as triplets and the load
 *        vector entry-by-entry, then finalises a compressed SparseMatrix. The
 *        element scatter loop and all conditions write through this object so
 *        the global stiffness is never materialised densely.
 *
 * @tparam T Scalar floating-point type.
 */
template <std::floating_point T>
class SystemAssembler
{
public:

    /**
     * @brief Construct an assembler for an n_dofs × n_dofs system.
     *
     * @param n_dofs       Total number of global DOFs (system dimension).
     * @param nnz_estimate Optional hint for the number of triplets to reserve.
     */
    explicit SystemAssembler(Index n_dofs, std::size_t nnz_estimate = 0)
        : n_(n_dofs),
          load_(Vector<T>::Zero(n_dofs))
    {
        if (nnz_estimate > 0) {
            triplets_.reserve(nnz_estimate);
        }
    }

    /// @brief Accumulate a single stiffness entry: K(i, j) += value.
    void add_stiffness(Index i, Index j, T value)
    {
        triplets_.emplace_back(i, j, value);
    }

    /// @brief Accumulate a single load entry: F(i) += value.
    void add_load(Index i, T value)
    {
        load_(i) += value;
    }

    /// @brief Mutable access to the assembled load vector.
    Vector<T>& load() { return load_; }

    /**
     * @brief Build the compressed global stiffness matrix from the accumulated
     *        triplets. Duplicate (i, j) entries are summed, reproducing the
     *        accumulation of a dense += scatter.
     */
    SparseMatrix<T> finalize_matrix()
    {
        SparseMatrix<T> K(n_, n_);
        K.setFromTriplets(triplets_.begin(), triplets_.end());
        K.makeCompressed();
        return K;
    }

private:

    /// @brief Total number of global DOFs (system dimension).
    Index n_;

    /// @brief Accumulated stiffness triplets, summed on finalisation.
    std::vector<Triplet<T>> triplets_;

    /// @brief Global load vector.
    Vector<T> load_;

};

} // namespace pyck

#endif // PYCK_SYSTEM_ASSEMBLER_HPP
