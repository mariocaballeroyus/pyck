#ifndef PYCK_CONDITION_HPP
#define PYCK_CONDITION_HPP

#include <cstddef>

#include "../assembly/dof_layout.hpp"
#include "../types.hpp"

namespace pyck
{

template <std::floating_point T>
class Condition
{
public:

    virtual ~Condition() = default;

    virtual std::size_t num_dofs() const { return 0; }

    virtual void allocate_dofs(DofLayout& layout,
                               DofLayout::BlockId primal_block) {}

    virtual void apply(Matrix<T>& stiffness,
                       Vector<T>& load,
                       const DofLayout& layout,
                       DofLayout::BlockId primal_block) const = 0;

};

} // namespace pyck

#endif // PYCK_CONDITION_HPP
