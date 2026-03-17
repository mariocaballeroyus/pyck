"""Linear-elastic assembly (stiffness + load)."""
 
from __future__ import annotations
 
from typing import TYPE_CHECKING, Sequence, Union, cast, Any
 
import numpy as np
import numpy.typing as npt
 
import pyck._pyck as _pyck
from pyck.constraints.direct_constraint import DirectConstraint
 
if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.conditions.condition import Condition
    from pyck.elements.element import Element
    from pyck.geometry.patch import Patch
 
 
class LinearElasticProblem:
    """Assemble global stiffness and load from per-element contributions.
 
    A problem contains a list of named patches and a single element
    formulation shared by all patches. Quadrature rules and conditions
    are assigned per-patch.
 
    Parameters
    ----------
    patches : Patch or Sequence[Patch]
        The geometry patches. Each must have a unique `name`.
    element : Element
        Finite element providing local stiffness.
    """
 
    def __init__(
        self,
        patches: Sequence[Patch],
        element: Element,
        quadrature: QuadratureRule,
    ) -> None:
        if len(patches) != 1:
            raise NotImplementedError(
                f"LinearElasticProblem currently supports exactly 1 patch, "
                f"got {len(patches)}."
            )
 
        self._patches: dict[str, Patch] = {}
        for p in patches:
            if p.name in self._patches:
                raise ValueError(f"Duplicate patch name: '{p.name}'")
            self._patches[p.name] = p
 
        self._element = element
        self._quadrature = quadrature
        self._conditions: dict[str, list[Condition]] = {
            name: [] for name in self._patches
        }
 
        self._cpp_objects: dict[str, Union[_pyck.LinearElasticProblem1d, _pyck.LinearElasticProblem2d]] = {}
        for name, patch in self._patches.items():
            if patch.tdim == 1:
                self._cpp_objects[name] = _pyck.LinearElasticProblem1d(
                    cast(_pyck.Patch1d, patch._cpp_object), 
                    cast(_pyck.Element1d, element._cpp_object),
                    cast(_pyck.QuadratureRule1d, quadrature._cpp_object)
                )
            elif patch.tdim == 2:
                self._cpp_objects[name] = _pyck.LinearElasticProblem2d(
                    cast(_pyck.Patch2d, patch._cpp_object), 
                    cast(_pyck.Element2d, element._cpp_object),
                    cast(_pyck.QuadratureRule2d, quadrature._cpp_object)
                )
            else:
                raise ValueError(f"Unsupported topological dimension: {patch.tdim}")
 
    @property
    def patch_names(self) -> list[str]:
        """Return the names of all patches in the problem."""
        return list(self._patches.keys())
 
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
        condition: Condition,
        *,
        patch: str | None = None,
    ) -> None:
        """Register a load/boundary condition for one or all patches.
 
        Parameters
        ----------
        condition : Condition
            A condition to be applied.
        patch : str, optional
            Name of the target patch. If None, applied to all patches.
        """
        for name in self._resolve_patch_names(patch):
            # Check for lazy-binding by inspecting _cpp_object attribute if it exists
            cpp_obj = getattr(condition, "_cpp_object", None)
            if cpp_obj is None:
                condition.bind(self._quadrature, self._element)
                cpp_obj = getattr(condition, "_cpp_object")
 
            self._conditions[name].append(condition)
            self._cpp_objects[name].add_condition(cast(Any, cpp_obj))
 
    def add_constraint(
        self,
        constraint: Any,
        *,
        patch: str | None = None,
    ) -> None:
        """Register a constraint for one or all patches.
 
        Parameters
        ----------
        constraint : Constraint
            A constraint to be applied.
        patch : str, optional
            Name of the target patch. If None, applied to all patches.
        """
        if isinstance(constraint, DirectConstraint):
            cpp = constraint._cpp_object
            for name in self._resolve_patch_names(patch):
                self._cpp_objects[name].add_direct_constraint(cpp)
        else:
            cpp = getattr(constraint, "_cpp_object", constraint)
            for name in self._resolve_patch_names(patch):
                self._cpp_objects[name].add_constraint(cpp)
 
    def assemble(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Assemble the global system.
 
        Returns
        -------
        K : ndarray, shape (n, n)
            Global stiffness matrix.
        F : ndarray, shape (n,)
            Global load vector.
        """
        # For now, single-patch only
        name = next(iter(self._cpp_objects))
        K, F = self._cpp_objects[name].assemble()
        return np.asarray(K), np.asarray(F).ravel()
 
    def __repr__(self) -> str:
        patches = ", ".join(f"'{n}'" for n in self._patches)
        return f"LinearElasticProblem(patches=[{patches}])"
 
 
def create_linear_elastic_problem(
    patches: Sequence[Patch],
    element: Element,
    quadrature: QuadratureRule,
) -> LinearElasticProblem:
    """Create a :class:`LinearElasticProblem` assembler.
 
    Parameters
    ----------
    patches : Sequence[Patch]
        Geometry on which the problem is defined.
    element : Element
        Finite element formulation.
    quadrature : QuadratureRule
        Quadrature rule for assembly.
 
    Returns
    -------
    LinearElasticProblem
    """
    return LinearElasticProblem(patches, element, quadrature)
