"""Linear-elastic assembly (stiffness + load)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.conditions.condition import Condition
    from pyck.elements.element import Element
    from pyck.geometry.curve_patch import CurvePatch


class LinearElasticProblem:
    """Assemble global stiffness and load from per-element contributions.

    Wraps :class:`pyck::LinearElasticProblem<double, 1>`.

    A problem contains a list of named patches and a single element
    formulation shared by all patches. Quadrature rules and conditions
    are assigned per-patch.

    Parameters
    ----------
    patches : sequence of CurvePatch
        The geometry patches.  Each must have a unique `name`.
        Currently only a single patch is supported.
    element : Element
        Finite element providing local stiffness.
    """

    def __init__(
        self,
        patches: Sequence[CurvePatch],
        element: Element,
    ) -> None:
        if len(patches) != 1:
            raise NotImplementedError(
                f"LinearElasticProblem currently supports exactly 1 patch, "
                f"got {len(patches)}."
            )

        # Build name → patch mapping
        self._patches: dict[str, CurvePatch] = {}
        for p in patches:
            if p.name in self._patches:
                raise ValueError(f"Duplicate patch name: '{p.name}'")
            self._patches[p.name] = p

        self._element = element

        # Per-patch state
        self._quadratures: dict[str, QuadratureRule] = {}
        self._conditions: dict[str, list[Condition]] = {
            name: [] for name in self._patches
        }

        # Build one C++ problem object per patch (only 1 for now)
        self._cpp_objects: dict[str, _pyck.LinearElasticProblem1D] = {}
        for name, patch in self._patches.items():
            self._cpp_objects[name] = _pyck.LinearElasticProblem1D(
                patch._cpp_object, element._cpp_object
            )

    # ------------------------------------------------------------------
    # Patch helpers
    # ------------------------------------------------------------------

    @property
    def patch_names(self) -> list[str]:
        """Return the names of all patches in the problem."""
        return list(self._patches.keys())

    def _resolve_patch_names(self, patch: str | None) -> list[str]:
        """Return the list of patch names targeted by *patch*.

        If *patch* is `None` the operation targets every patch.
        """
        if patch is None:
            return list(self._patches.keys())
        if patch not in self._patches:
            raise KeyError(
                f"Unknown patch '{patch}'. "
                f"Available patches: {list(self._patches.keys())}"
            )
        return [patch]

    # ------------------------------------------------------------------
    # Quadrature
    # ------------------------------------------------------------------

    def set_quadrature(
        self,
        quadrature: QuadratureRule,
        *,
        patch: str | None = None,
    ) -> None:
        """Set the quadrature rule for one or all patches.

        Parameters
        ----------
        quadrature : QuadratureRule
            Quadrature rule for numerical integration.
        patch : str or None
            Name of the target patch.  If `None` (default), the
            quadrature is applied to every patch.
        """
        for name in self._resolve_patch_names(patch):
            self._quadratures[name] = quadrature
            self._cpp_objects[name].set_quadrature(quadrature._cpp_object)

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def add_condition(
        self,
        condition: Condition,
        *,
        patch: str | None = None,
    ) -> None:
        """Register a load/boundary condition for one or all patches.

        If the condition is a lazy :class:`~pyck.conditions.LoadCondition`
        (created without a quadrature rule), it is finalised here using the
        target patch's quadrature rule before being passed to C++.

        Parameters
        ----------
        condition : Condition
            A condition, e.g. :class:`~pyck.conditions.LoadCondition` or a
            Dirichlet condition.
        patch : str or None
            Name of the target patch.  If `None` (default), the
            condition is added to every patch.
        """
        for name in self._resolve_patch_names(patch):
            if condition._cpp_object is None:
                quad = self._quadratures.get(name)
                if quad is None:
                    raise RuntimeError(
                        f"Cannot bind lazy condition to patch '{name}': "
                        "no quadrature rule has been set. "
                        "Call set_quadrature() first."
                    )
                condition.bind(quad, self._element)
            self._conditions[name].append(condition)
            self._cpp_objects[name].add_condition(condition._cpp_object)

    def add_constraint(
        self,
        constraint,
        *,
        patch: str | None = None,
    ) -> None:
        """Register a constraint for one or all patches.

        Parameters
        ----------
        constraint : Constraint
            A constraint, e.g. StrongDirichletConstraint.
        patch : str or None
            Name of the target patch.  If `None` (default), the
            constraint is added to every patch.
        """
        cpp = getattr(constraint, "_cpp_object", constraint)
        for name in self._resolve_patch_names(patch):
            self._cpp_objects[name].add_constraint(cpp)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Assemble the global system.

        Returns
        -------
        K : ndarray, shape (n, n)
            Global stiffness matrix.
        F : ndarray, shape (n,)
            Global load vector.
        """
        # For now, single-patch only — delegate directly
        name = next(iter(self._cpp_objects))
        K, F = self._cpp_objects[name].assemble()
        return np.asarray(K), np.asarray(F).ravel()

    def __repr__(self) -> str:
        patches = ", ".join(f"'{n}'" for n in self._patches)
        return f"LinearElasticProblem(patches=[{patches}])"


def create_linear_elastic_problem(
    patches: Sequence[CurvePatch] | CurvePatch,
    element: Element,
) -> LinearElasticProblem:
    """Create a :class:`LinearElasticProblem` assembler.

    Parameters
    ----------
    patches : CurvePatch or list of CurvePatch
        The geometry patches.  A single patch may be passed directly
        (it will be wrapped in a list).  Currently only a single patch
        is supported.
    element : Element
        Finite element providing local stiffness.

    Returns
    -------
    LinearElasticProblem
    """
    # Allow passing a bare patch for convenience
    if not isinstance(patches, (list, tuple)):
        patches = [patches]
    return LinearElasticProblem(patches, element)
