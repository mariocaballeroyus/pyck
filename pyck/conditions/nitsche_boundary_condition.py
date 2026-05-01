"""Nitsche-method boundary condition enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck
from pyck.conditions.boundary_field import FieldName, resolve_field

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch_boundary import PatchBoundary


class NitscheBoundaryCondition:
    """Nitsche method for weakly enforcing boundary fields.

    Stable, consistent, and symmetric Dirichlet enforcement without auxiliary
    unknowns. Bound to one boundary site (boundary, quadrature, thickness);
    each enforced field is added via :meth:`add` as a pair
    ``(displacement_field, traction_field)`` together with its penalty β and
    prescribed value. The effective penalty coefficient is β/h, where ``h`` is
    the ``thickness`` passed at construction.
    """

    _cpp_object: _pyck.NitscheBoundaryCondition2d | None

    def __init__(
        self,
        boundary: "PatchBoundary",
        thickness: float,
        quadrature: "QuadratureRule | None" = None,
    ) -> None:
        if not isinstance(boundary._cpp_object, _pyck.PatchBoundary2d):
            raise TypeError(
                f"NitscheBoundaryCondition requires a 2-D boundary patch, "
                f"got {type(boundary._cpp_object).__name__}."
            )
        if thickness <= 0.0:
            raise ValueError("NitscheBoundaryCondition: thickness must be positive.")

        self._boundary = boundary
        self._thickness = float(thickness)
        self._quadrature = quadrature
        self._terms: list[
            tuple[_pyck.BoundaryField, _pyck.BoundaryField, float, float]
        ] = []
        self._cpp_object = None

    def add(
        self,
        displacement_field: FieldName | _pyck.BoundaryField,
        traction_field: FieldName | _pyck.BoundaryField,
        penalty: float,
        value: float = 0.0,
    ) -> "NitscheBoundaryCondition":
        """Add a Nitsche term to enforce on this boundary.

        Parameters
        ----------
        displacement_field : str or BoundaryField
            Solution-side field (e.g. ``"w"``, ``"rot_n"``, ``"rot_s"``).
        traction_field : str or BoundaryField
            Traction-side field work-conjugate to the displacement
            (e.g. ``"q_n"``, ``"m_n"``, ``"m_ns"``).
        penalty : float
            Penalty parameter β; effective coefficient is β/h.
        value : float, optional
            Prescribed value (default 0.0).

        Returns
        -------
        NitscheBoundaryCondition
            Self, to allow chaining.
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        self._terms.append(
            (
                resolve_field(displacement_field),
                resolve_field(traction_field),
                float(penalty),
                float(value),
            )
        )
        return self

    def bind(self, _: "QuadratureRule", element: "Element") -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = self._quadrature if self._quadrature is not None else self._boundary.quadrature
        cpp = _pyck.NitscheBoundaryCondition2d(
            self._boundary._cpp_object,
            element._cpp_object,
            rule._cpp_object,
            self._thickness,
        )
        for displacement, traction, penalty, value in self._terms:
            cpp.add(displacement, traction, penalty, value)
        self._cpp_object = cpp

    def __repr__(self) -> str:
        return (
            f"NitscheBoundaryCondition(thickness={self._thickness!r}, "
            f"num_terms={len(self._terms)})"
        )


def create_clamped_nitsche(
    boundary: "PatchBoundary",
    thickness: float,
    penalty: float,
    quadrature: "QuadratureRule | None" = None,
) -> NitscheBoundaryCondition:
    """Create a clamped Nitsche condition (W = 0, ROT_N = 0, ROT_S = 0).

    Pairs each displacement field with its work-conjugate traction:
      * w     ↔  q_n
      * rot_n ↔  m_n
      * rot_s ↔  m_{ns}
    """
    cond = NitscheBoundaryCondition(boundary, thickness, quadrature)
    cond.add("w",     "q_n",  penalty, 0.0)
    cond.add("rot_n", "m_n",  penalty, 0.0)
    cond.add("rot_s", "m_ns", penalty, 0.0)
    return cond


def create_simply_supported_nitsche(
    boundary: "PatchBoundary",
    thickness: float,
    penalty: float,
    quadrature: "QuadratureRule | None" = None,
) -> NitscheBoundaryCondition:
    """Create a simply-supported Nitsche condition (W = 0, ROT_S = 0)."""
    cond = NitscheBoundaryCondition(boundary, thickness, quadrature)
    cond.add("w",     "q_n",  penalty, 0.0)
    cond.add("rot_s", "m_ns", penalty, 0.0)
    return cond


def create_displacement_nitsche(
    boundary: "PatchBoundary",
    thickness: float,
    penalty: float,
    value_w: float = 0.0,
    quadrature: "QuadratureRule | None" = None,
) -> NitscheBoundaryCondition:
    """Create a displacement-only Nitsche condition (W = value_w)."""
    cond = NitscheBoundaryCondition(boundary, thickness, quadrature)
    cond.add("w", "q_n", penalty, value_w)
    return cond
