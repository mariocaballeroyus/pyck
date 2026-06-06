#ifndef PYCK_PATCH_BLOCKS_HPP
#define PYCK_PATCH_BLOCKS_HPP

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>

#include "dof_layout.hpp"

namespace pyck
{

template <std::floating_point T, std::size_t d>
class Patch;

/**
 * @brief Maps each registered patch to its primal DOF block.
 *
 * Built and owned by `LinearElasticProblem` during assembly and passed by
 * const reference to every `Condition::apply`. A single-patch condition
 * resolves its own patch; a coupling condition resolves both of the patches it
 * ties together. Keying on patch identity (the pointer) means a condition never
 * needs to know the problem's patch ordering.
 *
 * @tparam T Scalar floating-point type.
 * @tparam d Patch / element parametric dimension.
 */
template <std::floating_point T, std::size_t d>
class PatchBlocks
{
public:

    /// @brief Register a patch's primal block.
    void add(const Patch<T, d>& patch, DofLayout::BlockId block)
    {
        map_.emplace(&patch, block);
    }

    /// @brief Primal block of a patch; throws if the patch is unregistered.
    DofLayout::BlockId primal(const Patch<T, d>& patch) const
    {
        const auto it = map_.find(&patch);
        if (it == map_.end()) {
            throw std::runtime_error(
                "PatchBlocks::primal: patch has no primal block "
                "(not registered with this problem).");
        }
        return it->second;
    }

private:

    /// @brief Patch identity to primal block id.
    std::unordered_map<const Patch<T, d>*, DofLayout::BlockId> map_;
};

} // namespace pyck

#endif // PYCK_PATCH_BLOCKS_HPP
