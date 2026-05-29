"""Linear-elastic assembly (stiffness + load)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Sequence, Union, cast, Any

import numpy as np
import numpy.typing as npt
from scipy import sparse

import pyck._pyck as _pyck
from pyck.conditions.condition import BindableCondition
from pyck.constraints.direct_constraint import DirectConstraint

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.conditions.condition import Condition
    from pyck.elements.element import Element
    from pyck.geometry.patch import Patch


class LinearElasticProblem:
    """Assemble global stiffness and load from per-element contributions.

    A problem contains a list of named patches that share a single element
    formulation and a single quadrature rule. Each patch occupies its own
    primal DOF block in the global system; conditions are routed to a
    specific patch via the ``patch`` keyword argument on
    :meth:`add_condition`.

    Parameters
    ----------
    patches : Patch or Sequence[Patch]
        The geometry patches. Each must have a unique ``name`` and all must
        share the same topological dimension.
    element : Element
        Finite element providing local stiffness, shared across patches.
    quadrature : QuadratureRule, optional
        Quadrature rule for assembly. Defaults to ``patches[0].quadrature``.
    """

    def __init__(
        self,
        patches: Sequence[Patch],
        element: Element,
        quadrature: QuadratureRule | None = None,
    ) -> None:
        if len(patches) == 0:
            raise ValueError("LinearElasticProblem requires at least one patch.")

        tdims = {p.tdim for p in patches}
        if len(tdims) > 1:
            raise ValueError(
                "All patches must share the same topological dimension; "
                f"got {sorted(tdims)}."
            )
        tdim = next(iter(tdims))

        self._patches: dict[str, Patch] = {}
        self._patch_idx: dict[str, int] = {}
        for i, p in enumerate(patches):
            if p.name in self._patches:
                raise ValueError(f"Duplicate patch name: '{p.name}'")
            self._patches[p.name] = p
            self._patch_idx[p.name] = i

        self._element = element
        if quadrature is None:
            quadrature = patches[0].quadrature
        self._quadrature = quadrature
        self._conditions: dict[str, list[Condition]] = {
            name: [] for name in self._patches
        }

        cpp_patches = [p._cpp_object for p in patches]
        cpp_elements = [element._cpp_object for _ in patches]
        cpp_quadratures = [quadrature._cpp_object for _ in patches]

        self._cpp_object: Union[_pyck.LinearElasticProblem1d, _pyck.LinearElasticProblem2d]
        if tdim == 1:
            self._cpp_object = _pyck.LinearElasticProblem1d(
                cast(Any, cpp_patches),
                cast(Any, cpp_elements),
                cast(Any, cpp_quadratures),
            )
        elif tdim == 2:
            self._cpp_object = _pyck.LinearElasticProblem2d(
                cast(Any, cpp_patches),
                cast(Any, cpp_elements),
                cast(Any, cpp_quadratures),
            )
        else:
            raise ValueError(f"Unsupported topological dimension: {tdim}")

    @property
    def element(self) -> Element:
        """Return the element formulation."""
        return self._element

    @property
    def patches(self) -> tuple[Patch, ...]:
        """Return the geometry patches in construction order."""
        return tuple(self._patches.values())

    @property
    def patch_names(self) -> list[str]:
        """Return the names of all patches in the problem."""
        return list(self._patches.keys())

    @property
    def num_physical_dofs(self) -> int:
        """Return the number of non-auxiliary solution DOFs."""
        num_node_dofs = int(self._element.num_node_dofs)
        return sum(
            patch.num_control_pts * num_node_dofs
            for patch in self._patches.values()
        )

    def _resolve_patch_names(self, patch: str | None = None) -> list[str]:
        """Return the list of patch names targeted by *patch*."""
        if patch is None:
            return list(self._patches.keys())
        if patch not in self._patches:
            raise KeyError(
                f"Unknown patch '{patch}'. "
                f"Available patches: {list(self._patches.keys())}"
            )
        return [patch]


    def add_condition(
        self,
        condition: Condition | Sequence[Condition],
        *,
        patch: str | None = None,
    ) -> None:
        """Register a load/boundary condition for one or all patches.

        Parameters
        ----------
        condition : Condition or sequence of Condition
            One condition, or several conditions to be applied.
        patch : str, optional
            Name of the target patch. If None, applied to all patches.
        """
        if isinstance(condition, (list, tuple)):
            for item in condition:
                self.add_condition(item, patch=patch)
            return

        cpp_obj = condition._cpp_object
        if cpp_obj is None:
            if not isinstance(condition, BindableCondition):
                raise TypeError(
                    f"Condition of type {type(condition).__name__} "
                    f"does not provide a bound C++ object or a bind() method."
                )
            condition.bind(self._quadrature, self._element)
            cpp_obj = condition._cpp_object
            if cpp_obj is None:
                raise TypeError(
                    f"Condition of type {type(condition).__name__} "
                    f"did not produce a C++ object after bind()."
                )

        for name in self._resolve_patch_names(patch):
            self._conditions[name].append(condition)

        # The C++ condition carries its own patch; registration infers the
        # target DOF block by identity, so register exactly once.
        self._cpp_object.add_condition(cast(Any, cpp_obj))

    def add_domain_load(
        self,
        value: npt.NDArray[np.floating]
        | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        *,
        patch: str | None = None,
    ) -> None:
        """Register a body force over a patch's interior.

        The body force is integrated during assembly inside the element loop. A
        native C++ functor evaluates at full speed; a Python callable re-enters
        the interpreter once per element.

        Parameters
        ----------
        value : ndarray or callable
            A uniform constant force per unit area ``(3,)``, or a callable
            mapping quadrature-point physical coordinates ``(n, 3)`` to force
            vectors ``(n, 3)``.
        patch : str, optional
            Target patch name. If None, the body force is applied to all patches.
        """
        for name in self._resolve_patch_names(patch):
            self._cpp_object.add_domain_load(
                cast(Any, self._patches[name]._cpp_object), value
            )

    def add_constraint(
        self,
        constraint: Any,
        *,
        patch: str | None = None,
    ) -> None:
        """Register a constraint on the global system.

        Constraints reference absolute global DOF indices in the unified
        layout and are applied after all element + condition assembly.
        The ``patch`` keyword is accepted for backward compatibility but is
        ignored: constraints act on the global DOF vector directly.
        """
        del patch  # constraints are global; argument retained for API stability
        cpp = getattr(constraint, "_cpp_object", constraint)
        self._cpp_object.add_constraint(cpp)

    def assemble(self) -> tuple[sparse.csc_matrix, npt.NDArray[np.float64]]:
        """Assemble the global system.

        Returns
        -------
        K : scipy.sparse.csc_matrix, shape (n, n)
            Global stiffness matrix.
        F : ndarray, shape (n,)
            Global load vector.
        """
        K, F = self._cpp_object.assemble()
        return K, np.asarray(F).ravel()

    def __repr__(self) -> str:
        patches = ", ".join(f"'{n}'" for n in self._patches)
        return f"LinearElasticProblem(patches=[{patches}])"


