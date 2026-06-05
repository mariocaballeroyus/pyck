"""Lagrange-multiplier boundary condition enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck
from pyck.conditions.condition import Field

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch_boundary import PatchBoundary


class LagrangeBoundaryCondition:
    """Boundary trace constraints enforced with Lagrange multipliers.

    A condition is bound to one boundary site (boundary, quadrature). Add as
    many fields as needed via :meth:`add`; each field gets its own multiplier
    DOF block, but the per-span shape evaluation is shared across all of them.
    """

    _cpp_object: _pyck.LagrangeBoundaryCondition2d | None

    def __init__(
        self,
        boundary: "PatchBoundary",
        quadrature: "QuadratureRule | None" = None,
    ) -> None:
        if not isinstance(boundary._cpp_object, _pyck.PatchBoundary2d):
            raise TypeError(
                f"LagrangeBoundaryCondition requires a 2-D boundary patch, "
                f"got {type(boundary._cpp_object).__name__}."
            )

        self._boundary = boundary
        self._quadrature = quadrature
        self._terms: list[tuple[_pyck.BoundaryValue, float]] = []
        self._cpp_object = None

    def add(
        self,
        field: Field,
        value: float = 0.0,
    ) -> "LagrangeBoundaryCondition":
        """Add a boundary value to enforce on this boundary.

        Parameters
        ----------
        field : Field
            Value to enforce (e.g. ``Field.U_X``, ``Field.ROT_N``).
        value : float, optional
            Prescribed value (default 0.0).

        Returns
        -------
        LagrangeBoundaryCondition
            Self, to allow chaining.
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        self._terms.append((_pyck.BoundaryValue(field), float(value)))
        return self

    def bind(self, _: "QuadratureRule", element: "Element") -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = self._quadrature if self._quadrature is not None else self._boundary.quadrature
        cpp = _pyck.LagrangeBoundaryCondition2d(
            self._boundary._cpp_object,
            element._cpp_object,
            rule._cpp_object,
        )
        for field, value in self._terms:
            cpp.add(field, value)
        self._cpp_object = cpp

    def __repr__(self) -> str:
        return f"LagrangeBoundaryCondition(num_fields={len(self._terms)})"


