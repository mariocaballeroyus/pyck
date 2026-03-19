"""Abstract base class for parametric patches.

Patches are the building blocks of isogeometric geometry. All patches in pyck
are embedded in 3D physical space; this dimension is fixed and cannot be reduced.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

import pyck._pyck as _pyck


class Patch(ABC):
    """Abstract base class for tensor-product parametric patches."""

    _cpp_object: _pyck.Patch1d | _pyck.Patch2d
    _name: str

    @property
    def name(self) -> str:
        """Patch label."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def gdim(self) -> int:
        """Geometric (physical) dimension."""
        return 3

    @property
    @abstractmethod
    def tdim(self) -> int:
        """Topological (parametric) dimension."""

    @cached_property
    def control_points(self) -> np.ndarray:
        """Read-only view of the control-point matrix, shape (n, 3)."""
        pts = np.asarray(self._cpp_object.control_pts())
        pts.flags.writeable = False
        return pts

    @property
    def num_control_pts(self) -> int:
        """Total number of control points."""
        return self._cpp_object.num_control_pts()

    def dof_mapper(self) -> _pyck.DofMapper1d | _pyck.DofMapper2d:
        """Return the DOF mapper for this patch."""
        return self._cpp_object.dof_mapper()