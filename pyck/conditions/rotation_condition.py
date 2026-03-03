"""Rotation boundary condition (zero rotation/slope)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.geometry.boundary_patch import BoundaryPatch


class RotationCondition:
    """Homogeneous rotation boundary condition on a patch boundary.

    Enforces zero rotation (slope) at the DOFs associated with a boundary patch.
    Currently only zero values are supported; passing a non-zero value raises
    a ``ValueError``.

    Parameters
    ----------
    boundary_patch : BoundaryPatch
        A boundary patch obtained via ``patch.boundary("start")`` or
        ``patch.boundary("end")``.
    value : float
        Prescribed rotation value (must be 0.0).
    """

    def __init__(
        self, boundary_patch: BoundaryPatch, *, value: float = 0.0,
    ) -> None:
        self._boundary_patch = boundary_patch
        self._value = float(value)

        dofs = sorted(set(boundary_patch.rotation_dofs))
        self._dofs = dofs

        try:
            self._cpp_object = _pyck.RotationCondition(dofs, self._value)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc

    def bind(self, quadrature) -> None:
        pass

    @property
    def boundary(self) -> str:
        """Which boundary: ``'start'`` or ``'end'``."""
        return self._boundary_patch.side

    @property
    def boundary_patch(self) -> BoundaryPatch:
        """The boundary patch this condition is applied to."""
        return self._boundary_patch

    @property
    def value(self) -> float:
        """Prescribed rotation value."""
        return self._value

    @property
    def dofs(self) -> list[int]:
        """Global DOF indices constrained by this condition."""
        return list(self._dofs)

    def __repr__(self) -> str:
        return (
            f"RotationCondition(boundary='{self._boundary_patch.side}', "
            f"θ={self._value})"
        )


def create_rotation_condition(
    bd_patch: BoundaryPatch,
    *,
    value: float = 0.0,
) -> RotationCondition:
    """Create a rotation boundary condition.

    Parameters
    ----------
    bd_patch : BoundaryPatch
        A boundary patch obtained via ``patch.boundary("start")`` or
        ``patch.boundary("end")``.
    value : float
        Prescribed rotation value (must be 0.0 for now).

    Returns
    -------
    RotationCondition
    """
    return RotationCondition(bd_patch, value=value)
