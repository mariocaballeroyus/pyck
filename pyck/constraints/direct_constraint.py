"""Python wrapper for strong Dirichlet boundary conditions."""

from __future__ import annotations

from typing import Sequence

import pyck._pyck as _pyck


class DirectConstraint:
    """Boundary condition that directly prescribes DOF values.

    Wraps :class:`pyck::DirectConstraint<double>`.

    Parameters
    ----------
    dofs : sequence of int
        Global DOF indices to constrain.
    value : float or sequence of float
        Prescribed value(s). A scalar is broadcast to all *dofs*.
    """

    _cpp_object: _pyck.DirectConstraint

    def __init__(self, dofs: Sequence[int], value: float | Sequence[float] = 0.0) -> None:
        self._dofs = list(dofs)
        if isinstance(value, (int, float)):
            self._values = [float(value)] * len(self._dofs)
            self._cpp_object = _pyck.DirectConstraint(self._dofs, float(value))
        else:
            self._values = [float(v) for v in value]
            self._cpp_object = _pyck.DirectConstraint(self._dofs, self._values)

    @property
    def dofs(self) -> list[int]:
        """Constrained DOF indices."""
        return self._dofs

    @property
    def values(self) -> list[float]:
        """Prescribed values."""
        return self._values

    def __repr__(self) -> str:
        return f"DirectConstraint(n_dofs={len(self._dofs)})"
