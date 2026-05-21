#ifndef PYCK_DOF_LAYOUT_HPP
#define PYCK_DOF_LAYOUT_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../types.hpp"

namespace pyck
{

/**
 * @brief Tag identifying the kind of unknown stored at a global DOF row.
 */
enum class DofType : std::uint8_t {
    Primal = 0,
    LagrangeMultiplier = 1,
};

/**
 * @brief Equation-numbering authority for assembly. 
 * 
 * Owns the global DOF numbering and a small descriptor for each allocated
 * contiguous DOF block: type, offset, size, and stride.
 */
class DofLayout
{
public:

    using BlockId = std::uint32_t;

    /**
     * @brief Descriptor of an allocated DOF block.
     */
    struct BlockInfo {
        DofType type;
        Index base;
        Index count;
        Index stride;
    };

    /**
     * @brief Allocate a block of DOFs.
     *
     * @param type    DofType::Primal or DofType::LagrangeMultiplier.
     * @param count   Total number of DOFs in the block.
     * @param stride  Number of DOFs associated to each control point.
     * @return        DOF block ID.
     */
    BlockId allocate(DofType type, Index count, Index stride)
    {
        const BlockId id = static_cast<BlockId>(blocks_.size());
        const Index base = total_;

        blocks_.push_back({type, base, count, stride});
        total_ += count;

        return id;
    }

    /**
     * @brief Reset the layout to an empty state (used between assemblies).
     */
    void clear()
    {
        blocks_.clear();
        total_ = 0;
    }

    /// @brief Tag of a block.
    DofType block_type(BlockId id) const { return blocks_[id].type; }

    /// @brief Total number of DOFs in a block.
    Index block_size(BlockId id) const { return blocks_[id].count; }

    /// @brief Total number of control points associated to a block.
    Index block_num_control_points(BlockId id) const { return blocks_[id].count / blocks_[id].stride; }

    /// @brief Base index of a block.
    Index block_base(BlockId id) const { return blocks_[id].base; }

    /// @brief Stride (dofs per control point) of a block.
    Index block_stride(BlockId id) const { return blocks_[id].stride; }

    /**
     * @brief Expand control-point indices to global DOF indices.
     *
     * Writes the expanded list into a caller-owned buffer (resized in place);
     * pre-sized workspaces (e.g. `PatchValues::elem_dofs`) make this
     * allocation-free.
     */
    void scatter_primal(BlockId block,
                        const std::vector<Index>& control_pts,
                        std::vector<Index>& out) const
    {
        const auto& b = blocks_[block];
        out.resize(control_pts.size() * static_cast<std::size_t>(b.stride));
        std::size_t w = 0;
        for (auto cp : control_pts) {
            const Index base_cp = b.base + cp * b.stride;
            for (Index v = 0; v < b.stride; ++v) {
                out[w++] = base_cp + v;
            }
        }
    }

    /// @brief Total number of DOFs declared so far.
    Index num_dofs() const { return total_; }

    /// @brief Number of allocated blocks.
    Index num_blocks() const { return blocks_.size(); }

private:

    /// @brief Allocated DOF blocks.
    std::vector<BlockInfo> blocks_;

    /// @brief Total number of DOFs.
    Index total_ = 0;
};

} // namespace pyck

#endif // PYCK_DOF_LAYOUT_HPP
