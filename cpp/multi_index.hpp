#ifndef PYCK_MULTI_INDEX_HPP
#define PYCK_MULTI_INDEX_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include "types.hpp"

namespace pyck
{

/**
 * @brief Number of distinct upper-triangular multi-indices of total order @p k
 *        on a @p d -dimensional parametric domain. Equals C(k + d - 1, d - 1).
 */
template <std::size_t d>
constexpr Index num_multi_indices(Index k)
{
    if (k == 0) return 1;
    if constexpr (d == 1) return 1;
    if constexpr (d == 2) return k + 1;
    if constexpr (d == 3) return (k + 1) * (k + 2) / 2;
    return 0;
}

/**
 * @brief Pack a symmetric (i, j) direction pair into a flat index using Voigt
 *        order: diagonals first (i == j → i), then off-diagonals (i < j) in
 *        lexicographic order.
 *
 *        d = 2: (0,0)→0, (1,1)→1, (0,1)→2
 *        d = 3: (0,0)→0, (1,1)→1, (2,2)→2, (0,1)→3, (0,2)→4, (1,2)→5
 */
template <std::size_t d>
constexpr Index pack2(Index i, Index j)
{
    if (i > j) { Index t = i; i = j; j = t; }
    if (i == j) return i;
    return d + (d - 1) * i - i * (i - 1) / 2 + (j - i - 1);
}

/**
 * @brief Pack a sorted-symmetric (i, j, k) triple into a flat index using
 *        lexicographic order over (i ≤ j ≤ k).
 *
 *        d = 2: (0,0,0)→0, (0,0,1)→1, (0,1,1)→2, (1,1,1)→3
 */
template <std::size_t d>
constexpr Index pack3(Index i, Index j, Index k)
{
    if (i > j) { Index t = i; i = j; j = t; }
    if (j > k) { Index t = j; j = k; k = t; }
    if (i > j) { Index t = i; i = j; j = t; }
    Index offset = 0;
    for (Index ii = 0; ii < i; ++ii)
        offset += (d - ii) * (d - ii + 1) / 2;
    for (Index jj = i; jj < j; ++jj)
        offset += (d - jj);
    return offset + (k - j);
}

/**
 * @brief Enumerate the direction-index tuples for derivatives of total order
 *        @p k in @p d directions, in the canonical pack order used by
 *        symmetric-tensor packed storage (BasisValues, IntrinsicGeometry, etc.).
 *
 *        k = 1: [{0}, {1}, …, {d-1}]
 *        k = 2: Voigt — diagonals first, then off-diagonals (i < j lex).
 *        k = 3: lex over (i ≤ j ≤ l).
 */
template <std::size_t d>
std::vector<std::vector<Index>>
enumerate_direction_indices(Index k)
{
    std::vector<std::vector<Index>> out;
    if (k == 0) {
        out.push_back({});
        return out;
    }
    if (k == 1) {
        for (Index i = 0; i < d; ++i) out.push_back({i});
        return out;
    }
    if (k == 2) {
        for (Index i = 0; i < d; ++i) out.push_back({i, i});
        for (Index i = 0; i < d; ++i)
            for (Index j = i + 1; j < d; ++j) out.push_back({i, j});
        return out;
    }
    if (k == 3) {
        for (Index i = 0; i < d; ++i)
            for (Index j = i; j < d; ++j)
                for (Index l = j; l < d; ++l) out.push_back({i, j, l});
        return out;
    }
    return out;
}

} // namespace pyck

#endif // PYCK_MULTI_INDEX_HPP
