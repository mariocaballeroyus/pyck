"""Penalty-based boundary condition enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck
from pyck.conditions.boundary_field import FieldName, resolve_field

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch_boundary import PatchBoundary


class PenaltyBoundaryCondition:
    """Penalty method for weakly enforcing boundary fields.

    A condition is bound to one boundary site (boundary, quadrature). Add as
    many fields as needed via :meth:`add`; the per-span shape evaluation is
    shared across all of them.
    """

    _cpp_object: _pyck.PenaltyBoundaryCondition2d | None

    def __init__(
        self,
        boundary: "PatchBoundary",
        quadrature: "QuadratureRule | None" = None,
    ) -> None:
        if not isinstance(boundary._cpp_object, _pyck.PatchBoundary2d):
            raise TypeError(
                f"PenaltyBoundaryCondition requires a 2-D boundary patch, "
                f"got {type(boundary._cpp_object).__name__}."
            )

        self._boundary = boundary
        self._quadrature = quadrature
        self._terms: list[tuple[_pyck.BoundaryField, float, float]] = []
        self._cpp_object = None

    def add(
        self,
        field: FieldName | _pyck.BoundaryField,
        penalty: float,
        value: float = 0.0,
    ) -> "PenaltyBoundaryCondition":
        """Add a field to enforce on this boundary.

        Parameters
        ----------
        field : str or BoundaryField
            Field to enforce (e.g. ``"w"``, ``"rot_n"``, ``"rot_s"``).
        penalty : float
            Penalty factor for this field.
        value : float, optional
            Prescribed value (default 0.0).

        Returns
        -------
        PenaltyBoundaryCondition
            Self, to allow chaining.
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        self._terms.append((resolve_field(field), float(penalty), float(value)))
        return self

    def bind(self, _: "QuadratureRule", element: "Element") -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = self._quadrature if self._quadrature is not None else self._boundary.quadrature
        cpp = _pyck.PenaltyBoundaryCondition2d(
            self._boundary._cpp_object,
            element._cpp_object,
            rule._cpp_object,
        )
        for field, penalty, value in self._terms:
            cpp.add(field, penalty, value)
        self._cpp_object = cpp

    def __repr__(self) -> str:
        return f"PenaltyBoundaryCondition(num_fields={len(self._terms)})"


def create_displacement_penalty(
    boundary: "PatchBoundary",
    quadrature: "QuadratureRule | None" = None,
    penalty: float = 1e12,
    value_w: float = 0.0,
) -> PenaltyBoundaryCondition:
    """Create a displacement penalty condition (W = value_w)."""
    cond = PenaltyBoundaryCondition(boundary, quadrature)
    cond.add("w", penalty, value_w)
    return cond


def create_simply_supported_penalty(
    boundary: "PatchBoundary",
    quadrature: "QuadratureRule | None" = None,
    penalty: float = 1e12,
) -> PenaltyBoundaryCondition:
    """Create a simply-supported penalty condition (W = 0, ROT_S = 0)."""
    cond = PenaltyBoundaryCondition(boundary, quadrature)
    cond.add("w", penalty, 0.0)
    cond.add("rot_s", penalty, 0.0)
    return cond


def create_clamped_penalty(
    boundary: "PatchBoundary",
    quadrature: "QuadratureRule | None" = None,
    penalty: float = 1e12,
) -> PenaltyBoundaryCondition:
    """Create a clamped penalty condition (W = 0, ROT_N = 0).

    Tangential rotation is not enforced explicitly: w = 0 along the entire
    edge already implies phi_s = -dw/dt = 0 along that edge, so a rot_s
    penalty term is redundant.
    """
    cond = PenaltyBoundaryCondition(boundary, quadrature)
    cond.add("w", penalty, 0.0)
    cond.add("rot_n", penalty, 0.0)
    return cond
