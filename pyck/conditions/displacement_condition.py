"""Displacement boundary condition (zero displacement)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.geometry.boundary_patch import BoundaryPatch


class DisplacementCondition:
    """Homogeneous displacement boundary condition on a patch boundary.

    Parameters
    ----------
    boundary_patch : BoundaryPatch
        A boundary patch obtained via ``patch.boundary("start")`` or
        ``patch.boundary("end")``.
    value : float
        Prescribed displacement value (must be 0.0).
    """

    def __init__(
        self, boundary_patch: BoundaryPatch, *, value: float = 0.0,
    ) -> None:
        self._boundary_patch = boundary_patch
        self._value = float(value)

        dofs = sorted(set(boundary_patch.displacement_dofs))
        self._dofs = dofs

        try:
            self._cpp_object = _pyck.DisplacementCondition(dofs, self._value)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc

    def bind(self, quadrature: QuadratureRule) -> None:
        # Dirichlet conditions do not depend on quadrature points
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
        """Prescribed displacement value."""
        return self._value

    @property
    def dofs(self) -> list[int]:
        """Global DOF indices constrained by this condition."""
        return list(self._dofs)

    def __repr__(self) -> str:
        return (
            f"DisplacementCondition(boundary='{self._boundary_patch.side}', "
            f"w={self._value})"
        )


def create_displacement_condition(
    bd_patch: BoundaryPatch,
    *,
    value: float = 0.0,
) -> DisplacementCondition:
    """Create a displacement boundary condition.

    Parameters
    ----------
    bd_patch : BoundaryPatch
        A boundary patch obtained via ``patch.boundary("start")`` or
        ``patch.boundary("end")``.
    value : float
        Prescribed displacement value (must be 0.0 for now).

    Returns
    -------
    DisplacementCondition
    """
    return DisplacementCondition(bd_patch, value=value)
